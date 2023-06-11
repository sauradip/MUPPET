# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transformer import SnippetEmbedding
import yaml
from scipy import ndimage
import itertools,operator
from config.dataset_class import activity_dict
from config.zero_shot import split_t1_train, split_t1_test, split_t2_train, split_t2_test , t1_dict_train , t1_dict_test , t2_dict_train , t2_dict_test
from config.few_shot import base_class,val_class,test_class,base_dict,val_dict,test_dict, base_train,base_train_dict
from MaskFormer.mask_former.modeling.transformer.transformer_predictor_v2 import TransformerPredictor
from transformers import CLIPTokenizer, CLIPModel
from transformers import CLIPTextModel, CLIPTextConfig
from AdaptFormer.models.vit_video_v2 import VisionTransformer
from functools import partial
from easydict import EasyDict
from collections import OrderedDict

from utils.backbone_utils import prepare_backbone

with open("./config/anet_2gpu.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

tun_config = EasyDict(
        # AdaptFormer
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=64,
        d_model=config['pretraining']['emb_dim'],
        # VPT related
        vpt_on=False,
        vpt_num=1,
    )

class STALE(nn.Module):
    def __init__(self):
        super(STALE, self).__init__()
        self.len_feat = config['model']['feat_dim']
        self.temporal_scale = config['model']['temporal_scale'] 
        self.split = config['dataset']['split']
        self.num_classes = config['dataset']['num_classes']+1
        self.n_heads = config['model']['embedding_head']
        self.emb_dim = config['pretraining']['emb_dim']
        self.embedding = SnippetEmbedding(self.n_heads, self.emb_dim, self.emb_dim, self.emb_dim, dropout=0.2)
        self.cross_att = SnippetEmbedding(self.n_heads, self.emb_dim, self.emb_dim, self.emb_dim, dq_model=self.emb_dim, dropout=0.3)
        self.query_adapt = SnippetEmbedding(self.n_heads, self.emb_dim, self.emb_dim, self.emb_dim, dropout=0.3)
        self.context_length = 30
        self.num_queries = 1
        self.txt_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.txt_model = self.txt_model.float()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.nshot = config['fewshot']['shot']
        self.cl_names = list(activity_dict.keys())
        self.delta = 0
        self.clip_stride = config['pretraining']['clip_stride']
        self.clip_max = config['pretraining']['clip_length']
        self.emb_dim = config['pretraining']['emb_dim']
        self.num_segment = int(self.clip_max / self.clip_stride)
        # self.act_prompt = self.get_prompt()
        self.bg_embeddings = nn.Parameter(
            torch.empty(1, 512)
        )
        self.proj = nn.Sequential(
            nn.Conv1d(self.emb_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.proj_txt = nn.Sequential(
            nn.Conv1d(512,self.emb_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )

        context_length = self.context_length
        self.contexts = nn.Parameter(torch.randn(1, context_length))
        nn.init.trunc_normal_(self.contexts)

        self.cls_adapter = nn.Sequential(
            nn.Conv1d(in_channels=self.emb_dim+self.emb_dim, out_channels=self.emb_dim, kernel_size=1,
            padding=0)
        )

        ##### ViT-B/16 Transformer Backbone ######
        self.video_backbone = VisionTransformer(
            # patch_size=16, 
            # embed_dim=768, 
            # depth=12, 
            # num_heads=12, 
            # mlp_ratio=4, 
            # qkv_bias=True,
            # norm_layer=partial(nn.LayerNorm, eps=1e-6),
            img_size=112, #224, 
            patch_size=16, 
            in_chans=3, 
            num_classes=1000, 
            embed_dim=self.emb_dim, # 768
            depth=1, # 12
            num_heads=4, 
            mlp_ratio=4., 
            qkv_bias=False, 
            qk_scale=None, 
            drop_rate=0., 
            attn_drop_rate=0.,
            drop_path_rate=0., 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            init_values=0.,
            use_learnable_pos_emb=False, 
            init_scale=0.,
            all_frames=self.clip_stride,
            tubelet_size=2,
            use_mean_pooling=True,
            tuning_config=tun_config
        ).cuda()

        # print(' - debug emb_dim:', self.emb_dim)
        # self.video_backbone = torch.nn.DataParallel(self.video_backbone).cuda() # better way is to load n_gpu from config
        self.backbone_ckpt = torch.load(config['pretraining']['video_transformer'], map_location='cpu')

        # self.video_backbone.load_state_dict()
        self.backbone_encoder = prepare_backbone(self.video_backbone,self.backbone_ckpt)

        self.masktrans = TransformerPredictor(
            in_channels=self.emb_dim,
            mask_classification=False,
            num_classes=self.num_classes,
            hidden_dim=self.emb_dim,
            num_queries=1,
            nheads=2,
            dropout=0.1,
            dim_feedforward=1,
            enc_layers=2,
            dec_layers=2,
            pre_norm=True,
            deep_supervision=False,
            mask_dim=self.emb_dim,
            enforce_input_project=True
        ).cuda()

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feat, out_channels=self.num_classes+1, kernel_size=1,
            padding=0)
        )
    
        self.bn1 = nn.BatchNorm1d(num_features=2048)
        self.localizer_mask = nn.Sequential(
            nn.Conv1d(in_channels=self.emb_dim, out_channels=256, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=self.temporal_scale, kernel_size=1,stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.reduce_mask = nn.Sequential(
            nn.Conv1d(100, 1, 1),
            nn.Sigmoid()
            # nn.Conv1d(1024, 512, 1)
        )

        self.mask_MLP = nn.Sequential(
            nn.Conv1d(5,1,1),
            nn.Sigmoid()
        )

    def get_prompt(self,cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append("a video of action"+" "+c)
        return temp_prompt

    def projection(self,loc_feat):
        proj_feat = self.proj(loc_feat)
        return proj_feat

    # def gen_text_query(self, vid_feat, mode):


    def compute_score_maps(self, visual, text):

        B,K,C = text.size()
        text_cls = text[:,:(K-1),:]
        text_cls = text_cls / text_cls.norm(dim=2, keepdim=True)
        text = text / text.norm(dim=2, keepdim=True)
        # visual = self.projection(visual)
        visual = torch.clamp(visual,min=1e-4)
        visual_cls = visual.mean(dim=2)
        visual = visual / visual.norm(dim=1, keepdim=True)
        visual_cls = visual_cls / visual_cls.norm(dim=1, keepdim=True)
        score_cls = torch.einsum('bc,bkc->bk', visual_cls, text_cls) * 100
        score_map = torch.einsum('bct,bkc->bkt', visual, text) * 100

        return score_map, score_cls

    def gen_text_query(self,vid_feat, mode):

        B,T,C = vid_feat.size()
        if self.nshot == 0:
            if mode == "train" and self.split == 50:
                cl_names = list(t2_dict_train.keys())
                self.num_classes = 100
            elif mode == "test" and self.split == 50:
                cl_names = list(t2_dict_test.keys())
                self.num_classes = 100
            elif mode == "train" and self.split == 75:
                cl_names = list(t1_dict_train.keys())
                self.num_classes = 150
            elif mode == "test" and self.split == 75:
                cl_names = list(t1_dict_test.keys())
                self.num_classes = 50
        else: 
            if mode == "train":
                cl_names = list(base_train_dict.keys())
                self.num_classes = 180
            elif mode == "test":
                cl_names = list(test_dict.keys())
                self.num_classes = 20
        
        act_prompt = self.get_prompt(cl_names)
        texts = self.tokenizer(act_prompt, padding=True, return_tensors="pt").to('cuda')
        text_cls = self.txt_model.get_text_features(**texts) ## [cls,txt_feat] --> [200,512]
        text_emb = torch.mean(text_cls,dim=0)
        # text_emb = torch.cat([text_cls,self.bg_embeddings],dim=0).expand(B,-1,-1)  ## [bs, cls+1 ,txt_feat] --> [bs,201,512]

        return text_emb


    def crop_features(self, feature, mask):

        dtype = mask.dtype
        trim_ten = []
        trim_feat = torch.zeros_like(feature)
        mask_fg = torch.ones_like(mask)
        mask_bg = torch.zeros_like(mask)
        for i in range(mask.size(0)):
            cls_thres = float(torch.mean(mask[i,:],dim=0).detach().cpu().numpy())
            top_mask = torch.where(mask[i,:] >= cls_thres, mask_fg[i,:], mask_bg[i,:]).cuda(0)
            top_loc = (top_mask==1).nonzero().squeeze().cuda(0)
            trim_feat[i,:,top_loc] = feature[i,:,top_loc]
            trim_ten.append(trim_feat)
        if len(trim_ten) == 0:
            trim_ten = feature
        else:
            trim_ten = torch.stack(trim_ten, dim=0)
    
        return trim_feat
        
    def text_features(self,vid_feat, mode):

        B,T,C = vid_feat.size()
        if self.nshot == 0:
            if mode == "train" and self.split == 50:
                cl_names = list(t2_dict_train.keys())
                self.num_classes = 100
            elif mode == "test" and self.split == 50:
                cl_names = list(t2_dict_test.keys())
                self.num_classes = 100
            elif mode == "train" and self.split == 75:
                cl_names = list(t1_dict_train.keys())
                self.num_classes = 150
            elif mode == "test" and self.split == 75:
                cl_names = list(t1_dict_test.keys())
                self.num_classes = 50
        else: 
            if mode == "train":
                cl_names = list(base_train_dict.keys())
                self.num_classes = 180
            elif mode == "test":
                cl_names = list(test_dict.keys())
                self.num_classes = 20
        
        act_prompt = self.get_prompt(cl_names)
        texts = self.tokenizer(act_prompt, padding=True, return_tensors="pt").to('cuda')
        text_cls = self.txt_model.get_text_features(**texts) ## [cls,txt_feat] --> [200,512]
        text_emb = torch.cat([text_cls,self.bg_embeddings],dim=0).expand(B,-1,-1)  ## [bs, cls+1 ,txt_feat] --> [bs,201,512]

        return text_emb


    def find_mask(self,raw_mask):
        seq = raw_mask.detach().cpu().numpy()
        # print(seq)
        m_th = np.mean(seq)
        filtered_seq = seq > 0.5
        integer_map = map(int,filtered_seq)
        filtered_seq_int = list(integer_map)
        filtered_seq_int2 = ndimage.binary_fill_holes(filtered_seq_int).astype(int).tolist()
        if 1 in filtered_seq_int2 :
            r = max((list(y) for (x,y) in itertools.groupby((enumerate(filtered_seq_int2)),operator.itemgetter(1)) if x == 1), key=len)
            if r[-1][0] - r[0][0] > 1:
                start_pt = r[0][0]
                end_pt = r[-1][0]
            else:
                start_pt = 0
                end_pt = 99
        else:
            start_pt = 0
            end_pt = 99
        return start_pt,end_pt


    def forward(self, snip, mode):
        # print(' - debug snip:', snip.device)
        ##### Extract the video features using Adaptformer #######
        raw_vid = snip
        snip_feat = [] 
        for i in range(self.num_segment):
            win_feat = self.backbone_encoder(raw_vid[:,:,self.clip_stride*i:self.clip_stride*i+self.clip_stride,:,:]) ## ip : [batch,channel=3,stride=16,H,W] op: [batch,emb_dim]
            snip_feat.append(win_feat)

        snip =  torch.cat(snip_feat,1).view(-1,self.num_segment,self.emb_dim) ## [batch,num_segment,emb_dim]
        snip = snip.permute(0,2,1)
        snip = F.interpolate(snip, size=self.temporal_scale, mode='linear',align_corners=False) ### interpolate temporal dimension to 100 ## [batch,100,emb_dim]

        vid_feature = snip
        snip = snip.permute(0,2,1)

        #### Temporal Embedding Module ##### 
        out = self.embedding(snip,snip,snip)
        out = out.permute(0,2,1)
        features = out


        fg_text_emb = self.gen_text_query(features,mode) ###[1, dim]
        fg_text_emb = self.proj_txt(fg_text_emb.unsqueeze(-1)).squeeze(-1)
        # print("text",fg_text_emb.size())
        fg_vid_emb = torch.mean(torch.mean(features,dim=2),dim=0) ### [1,dim]
        # print("vid", fg_vid_emb.size())
        fuse_emb = torch.cat([fg_vid_emb,fg_text_emb],dim=0).unsqueeze(0).unsqueeze(2)
        # print(fuse_emb.size())
        fg_query = self.cls_adapter(fuse_emb).squeeze(2).expand(self.num_queries,-1)

        
        ### Action Mask Localizer Branch ###
        
        bottom_br = self.localizer_mask(features)
        #### Representation Mask ####
        snipmask = self.masktrans(vid_feature.unsqueeze(2),features.unsqueeze(3),fg_query)
        bot_mask = torch.mean(bottom_br, dim=2)
        soft_mask = torch.sigmoid(snipmask["pred_masks"]).view(-1,self.temporal_scale)
        mask_feat = self.crop_features(features,bot_mask)
        soft_tensor = soft_mask
        text_feat = self.text_features(vid_feature, mode)
        text_feat = self.proj_txt(text_feat.permute(0,2,1))
        text_feat = text_feat.permute(0,2,1)
        mask_feat = mask_feat.permute(0,2,1)

        #### Vision-Language Cross-Adaptation ####
        # print(' - debug cross_att:', text_feat.shape, mask_feat.shape)
        text_feat_att = self.cross_att(text_feat, mask_feat, mask_feat)
        text_feat_fin = text_feat_att + self.delta*text_feat
        mask_feat = mask_feat.permute(0,2,1)
        score_maps, score_maps_class = self.compute_score_maps(mask_feat, text_feat_fin)

        ### Contextualized Vision-Language Classifier ###
        top_br = score_maps
        query_adapt_feat = self.query_adapt(features.permute(0,2,1), mask_feat.permute(0,2,1),mask_feat.permute(0,2,1))
        bottom_br = self.localizer_mask(query_adapt_feat.permute(0,2,1))

        return top_br, bottom_br , soft_tensor , score_maps_class, features
