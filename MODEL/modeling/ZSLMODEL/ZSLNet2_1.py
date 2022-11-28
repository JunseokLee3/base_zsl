import torch
import torch.nn as nn
import torch.nn.functional as F

from MODEL.modeling.backbone import resnet101_features,ViT
import MODEL.modeling.utils as utils

from os.path import join
import pickle
import numpy as np
import time
from MODEL.modeling.lossModule import SupConLoss_clear
from torch import distributed as dist
from .nets import *
from .class_name import *
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import os
from einops.layers.torch import Rearrange
import yaml
import numpy as np

from MODEL.data import build_dataloader

base_architecture_to_features = {
    'resnet101': resnet101_features,
}

# first I tried to attention 

class ZSLNet(nn.Module):
    def __init__(self, backbone, img_size, c, w, h,
                 attribute_num, cls_num, ucls_num, attr_group,attribute, w2v, dataset_name,config,
                 scale=20.0, device=None,cfg=None ):

        super(ZSLNet, self).__init__()
        ##      
        self.config = config
        ##

        self.device = device
        self.img_size = img_size
        self.attribute_num = attribute_num
        print('attribute_num',self.attribute_num)
        self.feat_channel = c
        self.feat_w = w
        self.feat_h = h
        print("TEMP",cfg.MODEL.LOSS.TEMP)
        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.cls_num = cls_num
        self.attr_group = attr_group
        self.att_assign = {}
        self.attribute = attribute
        self.w2v = w2v
        for key in attr_group:
            for att in attr_group[key]:
                self.att_assign[att] = key - 1


        if scale<=0:
            self.scale = nn.Parameter(torch.ones(1) * 25.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)

        res101_layers = list(backbone.children())
        self.backbone = nn.Sequential(*res101_layers[:-4])
        self.layer1 = res101_layers[-4]
        self.layer2 = res101_layers[-3]
        self.layer3 = res101_layers[-2]
        self.layer4 = res101_layers[-1]
       
        self.attr_proto_size = 2048
        self.part_num = self.attribute_num
        if self.attr_proto_size == self.feat_channel:
            self.fc_proto = nn.Identity()
        else:
            self.fc_proto = nn.Linear(self.feat_channel, self.attr_proto_size)
        # loss
        self.Reg_loss = nn.MSELoss()
        self.CLS_loss = nn.CrossEntropyLoss()
        self.iters = -1
        self.rank = dist.get_rank()
        self.dataset_name = dataset_name
        if dataset_name == 'CUB':
            self.class_names = CUB_CLASS
            self.attr_names = CUB_ATTRIBUTE
            hid_size = 1024
        if dataset_name == 'AwA2':
            self.class_names = AwA2_CLASS
            self.attr_names = AwA2_ATTRIBUTE
            hid_size = 1024
        if dataset_name == 'SUN':
            self.class_names = SUN_CLASS
            hid_size = 1024
        if dataset_name == 'APY':
            hid_size = 256
      
        if cfg.MODEL.ORTH:
            print('#'*100)
            print('orth')
            self.attribute_vector = nn.Parameter(nn.init.orthogonal_(torch.empty(self.part_num,self.part_num)),requires_grad=True)
        else:
            self.attribute_vector = nn.Parameter(torch.eye(self.part_num),requires_grad=False)

        self.memory_max_size = 1024
        out_channel = self.part_num
        #layers 
        out_channel1 = 300

        self.extract_1 =  torch.nn.Conv2d(256, out_channel1, kernel_size=8, stride=8)
        self.extract_2 = torch.nn.Conv2d(512, out_channel1, kernel_size=4, stride=4)
        self.extract_3 = torch.nn.Conv2d(1024, out_channel1, kernel_size=2, stride=2)
        self.extract_4 = torch.nn.Conv2d(2048, out_channel1, kernel_size=1, stride=1)
      
        self.contrastive_embedding = nn.Linear(self.attr_proto_size,cfg.MODEL.HID)
     

        nn.init.xavier_uniform_(self.extract_1.weight)
        nn.init.constant_(self.extract_1.bias,0)
        nn.init.xavier_uniform_(self.extract_2.weight)
        nn.init.constant_(self.extract_2.bias,0)
        nn.init.xavier_uniform_(self.extract_3.weight)
        nn.init.constant_(self.extract_3.bias,0)
        nn.init.xavier_uniform_(self.extract_4.weight)
        nn.init.constant_(self.extract_4.bias,0)
        self.proto_model = ProtoModel(self.part_num,hid_size,self.attr_proto_size,with_cn=True)
        self.atten_thr = cfg.MODEL.ATTEN_THR
        self.feat_memory = torch.empty(0,cfg.MODEL.HID).to(self.device)
        self.label_memory = torch.empty(0).to(self.device)
        self.contrast_loss = SupConLoss_clear(cfg.MODEL.LOSS.TEMP)
        self.alpha = cfg.MODEL.LOSS.ALPHA
        self.beta = cfg.MODEL.LOSS.BETA
        self.scale_semantic = cfg.MODEL.SCALE_SEMANTIC
        self.episilon = 0.1
        self.memory_max_size = 512
        #
        self.transzero = TransZero(config, self.attribute, self.w2v).to(self.device)



    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.backbone(x)
        return x

    def attr_decorrelation(self, query):
    
       loss_sum = 0

       for key in self.attr_group:
           group = self.attr_group[key]
           if query.ndim == 3:
               proto_each_group = query[:,group,:]  # g1 * v
               channel_l2_norm = torch.norm(proto_each_group, p=2, dim=1)
           else:
               proto_each_group = query[group, :]  # g1 * v
               channel_l2_norm = torch.norm(proto_each_group, p=2, dim=0)
           loss_sum += channel_l2_norm.mean()

       loss_sum = loss_sum.float()/len(self.attr_group)

       return loss_sum



    def attentionModule(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        N, C, W, H = x4.shape


        # seen_att_normalized.size() torch.Size([150, 312])
        parts_map = self.extract_1(x1) + self.extract_2(x2) + self.extract_3(x3) + self.extract_4(x4)
        # parts_map size: torch.Size([8, 312, 14, 14]) -> (8, 300, 14, 14)
       
        return  parts_map, x4
        
    def forward(self, x, att=None, label=None, seen_att=None,att_unseen=None):
        if att is not None:
            att[att < 0] = 0.
            if self.part_num > self.attribute_num:
                att = torch.cat((att,att.new_ones(len(att)).unsqueeze(1)*0.0),1)
            # att = F.normalize(att,dim=-1)
            att_binary = att.clone()
            att_binary[att_binary > 0] = 1.
        if seen_att is not None:
            if self.part_num > self.attribute_num:
                seen_att = torch.cat((seen_att,seen_att.new_ones(len(seen_att)).unsqueeze(1)*0.0),1)
        seen_att_normalized = F.normalize(seen_att,dim=-1)
        self.iters += 1
        feat = self.conv_features(x)
        parts_map, fs = self.attentionModule(feat)
        out_package = self.transzero(parts_map)
        visual_score = out_package['pred']
       
        if not self.training:
            return visual_score
        L_proto = torch.tensor(0).float().to(self.device)
        Lcls = torch.tensor(0).float().to(self.device)
        Lcls_att = torch.tensor(0).float().to(self.device)
        L_proto_align = torch.tensor(0).float().to(self.device)
        Lcpt = torch.tensor(0).float().to(self.device)
        Lreg = torch.tensor(0).float().to(self.device)
        Lad = torch.tensor(0).float().to(self.device)
    
        part_filter = self.att_weight&att_binary.bool()
        part_feats = F.normalize(part_feats, dim=-1)
        att_proto = self.proto_model(F.normalize(self.attribute_vector,-1) * np.sqrt(self.part_num), False)
        att_proto = F.normalize(att_proto,dim=-1)
       
        if part_filter.sum()>0:
            attr_proto_dist = torch.cat([cosine_distance(part_feat, att_proto).unsqueeze(0) for part_feat in part_feats],dim=0)
            index_pos = torch.arange(len(att_proto)).view(-1,1).expand(attr_proto_dist.size(0),-1,1).to(self.device)
            tmp = torch.arange(len(att_proto))
            index_neg = torch.cat([tmp[tmp!=i].unsqueeze(0) for i in range(len(att_proto))],dim=0).expand(attr_proto_dist.size(0),-1,-1).to(self.device)
            pos_dists = attr_proto_dist.gather(2,index_pos).squeeze()
            neg_dists = attr_proto_dist.gather(2,index_neg).squeeze()
            L_proto += F.relu(pos_dists - self.alpha*neg_dists.min(dim=-1)[0]  + self.beta)[part_filter].mean()

        if self.part_num > self.attribute_num:
            att[:,-1] = 0.1

        Lcls += self.CLS_loss(visual_score, label)
        Lcls_att += self.CLS_loss(semantic_score, label)
        part_feats = part_feats[part_filter]
        part_label = (torch.arange(self.part_num)[None, ...].repeat(len(att), 1).to(self.device))[part_filter].reshape(-1)
        if len(part_label)>0 and len(torch.unique(part_label)) != len(part_label):
            contrastive_embeddings = F.normalize(self.contrastive_embedding(part_feats),dim=-1)
         
            L_proto_align += self.contrast_loss(contrastive_embeddings,part_label)


           

        scale = self.scale.item()

        loss_dict = {
            'Reg_loss': Lreg,
            'Cls_loss': Lcls,
            'AD_loss': Lad,
            'CPT_loss': Lcpt,
            'ATTCLS_loss': Lcls_att,
            'Proto_loss': L_proto,
            'Proto_align_loss': L_proto_align,
            'scale': scale,
        }

        return loss_dict

    def CPT(self, atten_map):
       N, L, W, H = atten_map.shape
       xp = torch.tensor(list(range(W))).long().unsqueeze(1).to(self.device)
       yp = torch.tensor(list(range(H))).long().unsqueeze(0).to(self.device)

       xp = xp.repeat(1, H)
       yp = yp.repeat(W, 1)

       atten_map_t = atten_map.view(N, L, -1)
       value, idx = atten_map_t.max(dim=-1)

       tx = idx // H
       ty = idx - H * tx

       xp = xp.unsqueeze(0).unsqueeze(0)
       yp = yp.unsqueeze(0).unsqueeze(0)
       tx = tx.unsqueeze(-1).unsqueeze(-1)
       ty = ty.unsqueeze(-1).unsqueeze(-1)

       pos = (xp - tx) ** 2 + (yp - ty) ** 2

       loss = atten_map * pos

       loss = loss.reshape(N, -1).mean(-1)
       loss = loss.mean()

       return loss



class TransZero(nn.Module):
    def __init__(self, config, att, init_w2v_att,
                 is_bias=True, bias=1, is_conservative=True):
        super(TransZero, self).__init__()
        self.config = config
        self.dim_f = config.dim_f
        self.dim_v = config.dim_v
        self.nclass = config.num_class

        self.is_bias = False
        self.is_conservative = is_conservative
        # class-level semantic vectors
        self.att = nn.Parameter(F.normalize(att), requires_grad=False)
        # GloVe features for attributes name
        self.V = nn.Parameter(F.normalize(init_w2v_att), requires_grad=True)

        # mapping
        self.W_1 = nn.Parameter(nn.init.normal_(
            torch.empty(self.dim_v, config.tf_common_dim)), requires_grad=True)

            
        # transformer
        self.transformer = Transformer(
            ec_layer=config.tf_ec_layer,
            dc_layer=config.tf_dc_layer,
            dim_com=config.tf_common_dim,
            dim_feedforward=config.tf_dim_feedforward,
            dropout=config.tf_dropout,
            SAtt=config.tf_SAtt,
            heads=config.tf_heads,
            aux_embed=config.tf_aux_embed)
        # for loss computation
        self.log_softmax_func = nn.LogSoftmax(dim=1)
        self.weight_ce = nn.Parameter(torch.eye(self.nclass), requires_grad=False)
        self.CLS_loss = nn.CrossEntropyLoss()
        self.layer_se = nn.Sequential(nn.AdaptiveAvgPool1d(1),
            Rearrange('... () -> ...'),
            nn.Linear(config.tf_common_dim, self.nclass))

    def forward(self, input, from_img=False):
        Fs = input
        # transformer-based visual-to-semantic embedding
        v2s_embed,Trans_vis = self.forward_feature_transformer(Fs)
        # classification
        package = {'pred': self.forward_attribute(v2s_embed),
                   'embed': v2s_embed,
                   'Vis_feature' : Trans_vis}
        # Trans_vis size :  torch.Size([50, 196, 300])
        
        return package

    def forward_feature_transformer(self, Fs):
        # visual 
        if len(Fs.shape) == 4:
            shape = Fs.shape
            Fs = Fs.reshape(shape[0], shape[1], shape[2] * shape[3])
        Fs = F.normalize(Fs, dim=1)
        # attributes
        V_n = F.normalize(self.V) if self.config.normalize_V else self.V
        # locality-augmented visual features
        Trans_out, Trans_vis = self.transformer(Fs, V_n)
        # embedding to semantic space
        embed = torch.einsum('iv,vf,bif->bi', V_n, self.W_1, Trans_out)
  
        # Trans_vis size :  torch.Size([50, 196, 300])
        # V_n size :  torch.Size([312, 300])
        # self.W_1 size :  torch.Size([300, 300])
        # Trans_out size :  torch.Size([50, 312, 300])
        # embed : torch.size([50,300])

        return embed, Trans_vis.permute(0,2,1)

    def forward_attribute(self, embed):
        embed = torch.einsum('ki,bi->bk', self.att, embed)
        self.vec_bias = self.mask_bias*self.bias
        embed = embed + self.vec_bias
        return embed


    def compute_aug_cross_entropy_Vis(self, in_package):
        Labels = in_package['batch_label']
        S_pp_vis = in_package['Vis_feature']
        S_pp_vis = self.layer_se(S_pp_vis)
        if self.is_bias:
            S_pp_vis = S_pp_vis - self.vec_bias       

        Prob = self.log_softmax_func(S_pp_vis)

        loss = -torch.einsum('bk,bk->b', Prob, Labels)
        loss = torch.mean(loss)
        return loss

    def compute_aug_cross_entropy(self, in_package):
        Labels = in_package['batch_label']
        S_pp = in_package['pred']

        if self.is_bias:
            S_pp = S_pp - self.vec_bias


        Prob = self.log_softmax_func(S_pp)

        loss = -torch.einsum('bk,bk->b', Prob, Labels)
        loss = torch.mean(loss)
        return loss

    def compute_reg_loss(self, in_package):
        tgt = torch.matmul(in_package['batch_label'], self.att)
        embed = in_package['embed']
        loss_reg = F.mse_loss(embed, tgt, reduction='mean')
        return loss_reg

    def compute_loss(self, in_package):
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]
        
        loss_CE_Ve = self.compute_aug_cross_entropy_Vis(in_package)
        loss_CE = self.compute_aug_cross_entropy(in_package)
        loss_reg = self.compute_reg_loss(in_package)

        loss =   loss_CE  + self.config.lambda_reg * loss_reg + 0.1 * loss_CE_Ve
        out_package = {'loss': loss, 'loss_CE': loss_CE,
                        'loss_reg': loss_reg , 'loss_CE_Ve' : loss_CE_Ve}
        return out_package


class Transformer(nn.Module):
    def __init__(self, ec_layer=1, dc_layer=1, dim_com=300,
                 dim_feedforward=2048, dropout=0.1, heads=1,
                 in_dim_cv=2048, in_dim_attr=300, dim_com2=2048, SAtt=True,
                 aux_embed=True):
        super(Transformer, self).__init__()
        # input embedding
        self.embed_cv = nn.Sequential(nn.Linear(in_dim_cv, dim_com))
        if aux_embed:
            self.embed_cv_aux = nn.Sequential(nn.Linear(in_dim_cv, dim_com))
        self.embed_attr = nn.Sequential(nn.Linear(in_dim_attr, dim_com))
        # transformer encoder
        self.transformer_encoder = MultiLevelEncoder_woPad(N=ec_layer,
                                                           d_model=dim_com,
                                                           h=1,
                                                           d_k=dim_com,
                                                           d_v=dim_com,
                                                           d_ff=dim_feedforward,
                                                           dropout=dropout)
        # transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model=dim_com,
                                                nhead=heads,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                SAtt=SAtt)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=dc_layer)

    def forward(self, f_cv, f_attr):
        # linearly map to common dim
        h_cv = f_cv.permute(0, 2, 1)
        h_attr = self.embed_attr(f_attr)
        h_attr_batch = h_attr.unsqueeze(0).repeat(f_cv.shape[0], 1, 1)
        # visual encoder
        memory = self.transformer_encoder(h_cv, h_attr_batch).permute(1, 0, 2)
        # attribute-visual decoder
        out = self.transformer_decoder(h_attr_batch.permute(1, 0, 2), memory)
        return out.permute(1, 0, 2), memory.permute(1,0,2)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout,
                                                identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        self.mhatt2 = MultiHeadGeometryAttention_self(d_model, d_k, d_v, h, dropout,
                                                identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)

        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(
            d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values,l2, relative_geometry_weights,
                attention_mask=None, attention_weights=None, pos=None):
        q, k = (queries + pos, keys +
                pos) if pos is not None else (queries, keys)
        att = self.mhatt(q, k, values, relative_geometry_weights,
                         attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        att1 = self.mhatt2(att, l2, l2,
                         attention_mask, attention_weights)
        att2 = self.lnorm(att1 + self.dropout(att1))
        # att = att1+ att2

        ff = self.pwff(att2)
        return ff


class MultiLevelEncoder_woPad(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder_woPad, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        self.WGs = nn.ModuleList(
            [nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, input, l2, attention_mask=None, attention_weights=None, pos=None):
        relative_geometry_embeddings = BoxRelationalEmbedding(
            input, grid_size=(14, 14))
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(
            -1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [layer(
            flatten_relative_geometry_embeddings).view(box_size_per_head) for layer in self.WGs]
        relative_geometry_weights = torch.cat(
            (relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)
        out = input
        for layer in self.layers:
            out = layer(out, out, out,l2, relative_geometry_weights,
                        attention_mask, attention_weights, pos=pos)
        return out


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", SAtt=True):
        super(TransformerDecoderLayer, self).__init__(d_model, nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout,
                                                      activation=activation)
        self.SAtt = SAtt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt size :  torch.Size([312, 50, 300])
        # memory size :  torch.Size([196, 50, 300])
        if self.SAtt:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            # self.SAtt tgt2 size :  torch.Size([312, 50, 300])
            tgt = self.norm1(tgt + self.dropout1(tgt2))
            # print('self.SAtt tgt size : ', tgt.size())
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # self.SAtt tgt size :  torch.Size([312, 50, 300])
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def get_relative_pos(x, batch_size, norm_len):
    x = x.view(1, -1, 1).expand(batch_size, -1, -1)
    return x / norm_len


def get_grids_pos(batch_size, seq_len, grid_size=(7, 7)):
    assert seq_len == grid_size[0] * grid_size[1]
    x = torch.arange(0, grid_size[0]).float().cuda()
    y = torch.arange(0, grid_size[1]).float().cuda()
    px_min = x.view(-1, 1).expand(-1, grid_size[0]).contiguous().view(-1)
    py_min = y.view(1, -1).expand(grid_size[1], -1).contiguous().view(-1)
    px_max = px_min + 1
    py_max = py_min + 1
    rpx_min = get_relative_pos(px_min, batch_size, grid_size[0])
    rpy_min = get_relative_pos(py_min, batch_size, grid_size[1])
    rpx_max = get_relative_pos(px_max, batch_size, grid_size[0])
    rpy_max = get_relative_pos(py_max, batch_size, grid_size[1])
    return rpx_min, rpy_min, rpx_max, rpy_max


def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True,
                           grid_size=(7, 7)):
    batch_size, seq_len = f_g.size(0), f_g.size(1)
    x_min, y_min, x_max, y_max = get_grids_pos(batch_size, seq_len, grid_size)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)
    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)
    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))
    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)
    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))
        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(
            batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat
        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)


class ScaledDotProductGeometryAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        super(ScaledDotProductGeometryAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()
        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, box_relation_embed_matrix,
                attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h,
                                    self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h,
                                   self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        w_g = box_relation_embed_matrix
        w_a = att
        w_mn = - w_g + w_a
        w_mn = torch.softmax(w_mn, -1)
        att = self.dropout(w_mn)
        out = torch.matmul(att, v).permute(
            0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out

class ScaledDotProductGeometryAttention_self(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        super(ScaledDotProductGeometryAttention_self, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()
        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, 
                attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h,
                                    self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h,
                                   self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        
        w_a = att
        w_mn =  w_a
        w_mn = torch.softmax(w_mn, -1)
        att = self.dropout(w_mn)
        out = torch.matmul(att, v).permute(
            0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class MultiHeadGeometryAttention_self(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False,
                 can_be_stateful=False, attention_module=None,
                 attention_module_kwargs=None, comment=None):
        super(MultiHeadGeometryAttention_self, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductGeometryAttention_self(
            d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values,
                attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values
        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, 
                                 attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, 
                                 attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


class MultiHeadGeometryAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False,
                 can_be_stateful=False, attention_module=None,
                 attention_module_kwargs=None, comment=None):
        super(MultiHeadGeometryAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductGeometryAttention(
            d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, relative_geometry_weights,
                attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values
        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, relative_geometry_weights,
                                 attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, relative_geometry_weights,
                                 attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out



def build_ZSLNet(cfg):
    dataset_name = cfg.DATASETS.NAME
    info = utils.get_attributes_info(dataset_name)
    attribute_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]

    attr_group = utils.get_attr_group(dataset_name)

    img_size = cfg.DATASETS.IMAGE_SIZE

    cofig_path = 'config/cub_gzsl.yaml'
    with open('CUB_GZSL.yaml') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    c,w,h = 2048, img_size//32, img_size//32

    scale = cfg.MODEL.SCALE

    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    model_dir = cfg.PRETRAINED_MODELS

    res101 = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    # vit = ViT(model_name='vit_large_patch16_224')

    w2v_file = dataset_name+"_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)
    _, _, _, res = build_dataloader(cfg, is_distributed=True)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)
    

    device = torch.device(cfg.MODEL.DEVICE)
    attribute = res['attribute'].to(device)

    return ZSLNet(backbone=res101, img_size=img_size,
                  c=c, w=w, h=h, scale=scale,
                  attribute_num=attribute_num,
                  attr_group=attr_group, attribute = attribute, w2v=w2v,dataset_name=dataset_name,config=config,
                  cls_num=cls_num, ucls_num=ucls_num,
                  device=device,cfg=cfg)