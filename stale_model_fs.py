# # -*- coding: utf-8 -*-

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
# from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from config.dataset_class import activity_dict
from config.zero_shot import split_t1_train, split_t1_test, split_t2_train, split_t2_test , t1_dict_train , t1_dict_test , t2_dict_train , t2_dict_test
from config.few_shot import base_class,val_class,test_class,base_dict,val_dict,test_dict, base_train,base_train_dict
# from denseclip.untils import tokenize
# from denseclip.models import Transformer,LayerNorm
# import denseclip
from MaskFormer.mask_former.modeling.transformer.transformer_predictor_v2 import TransformerPredictor
from transformers import CLIPTokenizer, CLIPModel
from transformers import CLIPTextModel, CLIPTextConfig
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from utils.backbone_utils import prepare_backbone
from AdaptFormer.models.vit_video_v2 import VisionTransformer
from functools import partial
from easydict import EasyDict
# from config.dataset_class import activity_dict, thumos_dict , thumos_dict2



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
        self.clip_stride = config['pretraining']['clip_stride']
        self.clip_max = config['pretraining']['clip_length']
        self.emb_dim = config['pretraining']['emb_dim']
        self.num_segment = int(self.clip_max / self.clip_stride)
        # self.embedding = SnippetEmbedding(self.n_heads, self.emb_dim, self.len_feat, self.len_feat, dropout=0.2)
        # self.cross_att = SnippetEmbedding(self.n_heads, self.emb_dim, self.emb_dim, self.emb_dim, dq_model=self.len_feat, dropout=0.3)
        self.embedding = SnippetEmbedding(self.n_heads, self.emb_dim, self.emb_dim, self.emb_dim, dropout=0.2)
        self.cross_att = SnippetEmbedding(self.n_heads, self.emb_dim, self.emb_dim, self.emb_dim, dq_model=self.emb_dim, dropout=0.3)
        self.query_adapt = SnippetEmbedding(self.n_heads, self.emb_dim, self.emb_dim, self.emb_dim, dropout=0.3)
        self.context_length = 30
        self.txt_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").float()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.nshot = config['fewshot']['shot']
        self.cl_names = list(activity_dict.keys())
        self.contxt_token = config['fewshot']['num_context']
        self.supp_trim = config['fewshot']['trim_support']
        self.delta = 0
        self.num_queries = 1
        self.bg_embeddings = nn.Parameter(
            torch.empty(1, 512)
        )

        self.proj = nn.Sequential(
            nn.Conv1d(self.emb_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.proj_txt = nn.Sequential(
            nn.Conv1d(512,768, kernel_size=1,padding=0)
        )

        context_length = self.context_length
        self.contexts = nn.Parameter(torch.randn(1, context_length))
        nn.init.trunc_normal_(self.contexts)

        ##### ViT-B/16 Transformer Backbone ######
        self.video_backbone = VisionTransformer(
            # patch_size=16, 
            # embed_dim=768, 
            # depth=12, 
            # num_heads=12, 
            # mlp_ratio=4, 
            # qkv_bias=True,
            # norm_layer=partial(nn.LayerNorm, eps=1e-6),
            img_size=112, 
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

        # self.video_backbone = torch.nn.DataParallel(self.video_backbone, device_ids=[0]).cuda()
        self.backbone_ckpt = torch.load(config['pretraining']['video_transformer'], map_location='cpu')

        # self.video_backbone.load_state_dict()
        self.backbone_encoder = prepare_backbone(self.video_backbone,self.backbone_ckpt)
        # self.backbone_encoder = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        self.masktrans = TransformerPredictor(
            in_channels=768,
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
    
        self.cls_adapter = nn.Sequential(
            nn.Conv1d(in_channels=self.emb_dim+self.emb_dim, out_channels=self.emb_dim, kernel_size=1,
            padding=0)
        )

        self.bn1 = nn.BatchNorm1d(num_features=2048)
        self.localizer_mask = nn.Sequential(
            nn.Conv1d(in_channels=self.emb_dim, out_channels=256, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=self.temporal_scale, kernel_size=1,stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.avg_pool = torch.nn.AdaptiveAvgPool1d(self.contxt_token)

        self.reduce_mask = nn.Sequential(
            nn.Conv1d(100, 1, 1),
            nn.Sigmoid()
        )

        self.mask_MLP = nn.Sequential(
            nn.Conv1d(5,1,1),
            nn.Sigmoid()
        )

    def get_prompt(self,cl_names,cl_dict):
        temp_prompt = []
        label_id = []
        meta_learn_class = config['fewshot']['meta_class']
        for c in cl_names:
            if meta_learn_class:
                temp_prompt.append(c)
                label_id.append(cl_dict[c])
            else:
                temp_prompt.append("a video of action"+" "+c)
                label_id.append(cl_dict[c])

        return temp_prompt, label_id

    def projection(self,loc_feat):
        proj_feat = self.proj(loc_feat)
        return proj_feat

    def compute_score_maps(self, visual, text):

        B,K,C = text.size()
        text_cls = text[:,:(K-1),:]
        text_cls = text_cls / text_cls.norm(dim=2, keepdim=True)
        text = text / text.norm(dim=2, keepdim=True)
        visual = torch.clamp(visual,min=1e-4)
        visual_cls = visual.mean(dim=2)
        visual = visual / visual.norm(dim=1, keepdim=True)
        visual_cls = visual_cls / visual_cls.norm(dim=1, keepdim=True)
        score_cls = torch.einsum('bc,bkc->bk', visual_cls, text_cls) * 100
        score_map = torch.einsum('bct,bkc->bkt', visual, text) * 100

        return score_map, score_cls

    

    def crop_features(self, feature, mask):
        # print(feature.size(),mask.size())
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


    def save_plots(self,embedding,test_dict):

        key_list = list(test_dict.keys())
        val_list = list(test_dict.values())

        _, label_list = self.get_prompt(test_dict.keys(),test_dict)

        text_cls = embedding.detach().cpu().numpy()
        fashion_tsne = TSNE(random_state=123).fit_transform(text_cls)

        print('t-SNE done!')

        x = fashion_tsne 
        colors = np.asarray(label_list)
        num_classes = len(np.unique(colors))
        palette = np.array(sns.color_palette("hls", num_classes))

        # create a scatter plot.
        plt.figure(figsize=(16, 16))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], lw=20, s=60, c=palette[colors.astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')


        # add the labels for each digit corresponding to the label
        txts = []

        for i in range(num_classes):

            # Position of each label at median of data points.

            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, key_list[val_list.index(i)] , fontsize=13)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
            
        plt.savefig('tsne.png')


    def gen_text_query(self,vid_feat,sub_cls,sub_proto, mode):
        B,T,C = vid_feat.size()
        if len(sub_cls) == 0:
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
        else:
            cl_names = sub_cls
            self.num_classes = len(sub_cls)

        act_prompt,_ = self.get_prompt(cl_names,test_dict)
        texts = self.tokenizer(act_prompt, padding=True, return_tensors="pt").to('cuda') 
        text_cls = self.txt_model.get_text_features(**texts) ## [cls,txt_feat] --> [200,512]
        text_emb = torch.mean(text_cls,dim=0)

        return text_emb
        
    def text_features(self,vid_feat, sub_cls, sup_proto, mode):
        B,T,C = vid_feat.size()
        if len(sub_cls) == 0:
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
        else:
            cl_names = sub_cls
            self.num_classes = len(sub_cls)
        
        sub_cls_dict = {}
        nway = self.num_classes
        nshot = int(B/nway)
        act_prompt,_ = self.get_prompt(cl_names,test_dict)
        # print(act_prompt)
        meta_learn_class = config['fewshot']['meta_class']


        if meta_learn_class :

            texts = self.tokenizer(act_prompt, padding=True, return_tensors="pt").to('cuda')
            cls_text_emb = texts["input_ids"]
            # print(sup_proto.size())
            context = sup_proto.squeeze(1).type(torch.LongTensor).to('cuda')
            context = context.expand(len(sub_cls),-1)
            # print(context.shape, cls_text_emb.shape, ) # torch.Size([1, 20]) torch.Size([5, 5])
            # cls_text_emb = cls_text_emb.mean(dim=0)
            text_context = torch.cat([context,cls_text_emb],dim=1)
            # print("successful",text_context.size())
            text_cls = self.txt_model.get_text_features(text_context) ## [cls,txt_feat] --> [200,512]
            text_emb = torch.cat([text_cls,self.bg_embeddings],dim=0).expand(B,-1,-1)  ## [bs, cls+1 ,txt_feat] --> [bs,201,512]
        else:
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
        # print(start_pt,end_pt)
        return start_pt,end_pt


    def forward(self, sup_snip, query_snip, sub_cls, sup_proto_gt, sup_trim, mode):

       ##### Extract the video features using Adaptformer #######
        raw_vid = sup_snip
        snip_feat = [] 
        for i in range(self.num_segment):
            win_feat = self.backbone_encoder(raw_vid[:,:,self.clip_stride*i:self.clip_stride*i+self.clip_stride,:,:]) ## ip : [batch,channel=3,stride=16,H,W] op: [batch,emb_dim]
            snip_feat.append(win_feat)

        snip =  torch.cat(snip_feat,1).view(-1,self.num_segment,self.emb_dim) ## [batch,num_segment,emb_dim]
        snip = snip.permute(0,2,1)
        sup_snip = F.interpolate(snip, size=self.temporal_scale, mode='linear',align_corners=False) ### interpolate temporal dimension to 100 ## [batch,100,emb_dim]
        vid_feature = sup_snip

        if mode == "test":
            raw_vid_alt = query_snip
            snip_feat_q = [] 
            for i in range(self.num_segment):
                win_feat = self.backbone_encoder(raw_vid_alt[:,:,self.clip_stride*i:self.clip_stride*i+self.clip_stride,:,:]) ## ip : [batch,channel=3,stride=16,H,W] op: [batch,emb_dim]
                snip_feat_q.append(win_feat)

            snip_alt =  torch.cat(snip_feat_q,1).view(-1,self.num_segment,self.emb_dim) ## [batch,num_segment,emb_dim]
            snip_alt = snip_alt.permute(0,2,1)
            q_snip = F.interpolate(snip_alt, size=self.temporal_scale, mode='linear',align_corners=False) ### interpolate temporal dimension to 100 ## [batch,100,emb_dim]
            vid_feature_alt = q_snip


        
        # print(mode,)
        B,C,T = vid_feature.size()

        snip = sup_snip.permute(0,2,1)
        out = self.embedding(snip,snip,snip)
        out = out.permute(0,2,1)
        features = out

        if mode == "train":
            nway,nshot,feat_dim,temp_dim = sup_proto_gt.size()
            sup_proto_trimmed = vid_feature.view(nway,nshot,feat_dim,temp_dim)*sup_proto_gt
            sup_proto = torch.mean(sup_proto_trimmed,dim=1) ## nshot consumed
            sup_proto = torch.mean(sup_proto, dim = 2) ## tmp_dim consumed
            # print("from support proto", sup_proto.size()) ## 1 x 768
        else:

            # print(' -- debug', sup_proto_gt.shape)
            nway,nshot,feat_dim,temp_dim = sup_proto_gt.size()
            # print(' -- debug', vid_feature_alt.shape)
            sup_proto_trimmed = vid_feature_alt.view(nway,nshot,feat_dim,temp_dim)*sup_proto_gt
            sup_proto = torch.mean(sup_proto_trimmed,dim=1) ## nshot consumed
            sup_proto = torch.mean(sup_proto, dim = 2) ## tmp_dim consumed
            # print("from query proto", sup_proto.size())

        #### Vision-Language fusion query #### 
        fg_text_emb = self.gen_text_query(features,sub_cls,sup_proto,mode) ###[1, dim]
        # print("text",fg_text_emb.size())
        fg_text_emb = self.proj_txt(fg_text_emb.unsqueeze(-1)).squeeze(-1)
        # print("text",fg_text_emb.size())
        # fg_vid_emb = torch.mean(torch.mean(features,dim=2),dim=0) ### [1,dim]
        fg_vid_emb = torch.mean(torch.mean(features,dim=2),dim=0) ### [1,dim]
        # print("vid", fg_vid_emb.size())
        fuse_emb = torch.cat([fg_vid_emb,fg_text_emb],dim=0).unsqueeze(0).unsqueeze(2)
        # print(fuse_emb.size())
        fg_query = self.cls_adapter(fuse_emb).squeeze(2).expand(self.num_queries,-1)

        ### Action Mask Localizer Branch ###
        if mode == "train":
            bottom_br = self.localizer_mask(features)
        else:
            bottom_br = self.localizer_mask(features)

        #### Few-Shot Specific #####
        sup_context = self.avg_pool(sup_proto.unsqueeze(1)) ## 1 x 20

        #### Representation Mask ####
        snipmask = self.masktrans(vid_feature.unsqueeze(2),features.unsqueeze(3),fg_query)
        bot_mask = torch.mean(bottom_br, dim=2)
        soft_mask = torch.sigmoid(snipmask["pred_masks"]).view(-1,self.temporal_scale)
        mask_feat = self.crop_features(features,soft_mask)
        soft_tensor = bot_mask
        text_feat = self.text_features(features, sub_cls, sup_context, mode)

        text_feat = self.proj_txt(text_feat.permute(0,2,1)) ## 512 to 768 projection
        text_feat = text_feat.permute(0,2,1)

        mask_feat = mask_feat.permute(0,2,1)

        #### Vision-Language Cross-Adaptation ####
        text_feat_att = self.cross_att(text_feat, mask_feat, mask_feat)
        text_feat_fin = text_feat_att + text_feat
        mask_feat = mask_feat.permute(0,2,1)
        score_maps, score_maps_class = self.compute_score_maps(mask_feat, text_feat_fin)
        
        #### Contextualized Vision-Language Classifier #####
        top_br = score_maps
        # print(top_br)
        query_adapt_feat = self.query_adapt(features.permute(0,2,1), mask_feat.permute(0,2,1),mask_feat.permute(0,2,1))
        bottom_br = self.localizer_mask(query_adapt_feat.permute(0,2,1))
        # print(mode, top_br.size(), bottom_br.size() , soft_tensor.size() , score_maps_class.size(), features.size())

        return top_br, bottom_br , soft_tensor , score_maps_class, features



    def extract_feature(self, snip):

        vid_feature = snip
        snip = snip.permute(0,2,1)
        out = self.embedding(snip,snip,snip)
        out = out.permute(0,2,1)
        features = out

        snipmask = self.masktrans(vid_feature.unsqueeze(2),features.unsqueeze(3))

        ### Global Segmentation Mask Branch ###
        bottom_br = self.global_mask(features)

        bot_mask = torch.mean(bottom_br, dim=1)


        soft_mask = torch.sigmoid(snipmask["pred_masks"]).view(-1,self.temporal_scale)


        # mask_feat = self.crop_features(features,soft_mask)
        mask_feat = self.crop_features(features,soft_mask)

        # soft_tensor = bot_mask
        soft_tensor = soft_mask

        text_feat = self.text_features(vid_feature, mode)
        # print(mask_feat)

        return mask_feat , text_feat



# ############## Sanity Check ##############

# print("executed-here")
# output_path = config['dataset']['testing']['output_path']
# ## few-shot setting ##
# fsmode = config['fewshot']['mode']
# nepisode = config['fewshot']['episode']
# nshot = config['fewshot']['shot']
# is_postprocess = True
# if not os.path.exists(output_path + "/results"):
#     os.makedirs(output_path + "/results")

# ### Load Model ###
# model = STALEFS()
# model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
# ### Load Base Class trained Model Checkpoint ###
# # print(' -- debug', output_path + "/STALE_base_best.pth.tar")
# # checkpoint = torch.load(output_path + "/STALE_base_best.pth.tar")
# # model.load_state_dict(checkpoint['state_dict'])
# model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
# # print('use {} gpus to train!'.format(opt['n_gpu']))

# # test_loader = torch.utils.data.DataLoader(stale_dataset_fs.STALEEpisodicDataset(subset="validation", mode='inference'),
# #                                             batch_size=1, shuffle=False,
# #                                             num_workers=8, pin_memory=False, drop_last=False)

# query_data = torch.rand(2,3,768,112,112) ## [nway,C,H,W] --> no shot for query
# support_data = torch.rand(4,3,768,112,112) ## [nway,nshot,C,H,W]
# supp_cls = ['Hurling', 'Polishing forniture'] ## [random n-way class list len(nway)]
# support_proto = torch.rand(2,2,768,100) # [nway,nshot,C,T]
# support_trimmed = torch.rand(2,2,768,100) ### not used in model file , so doesnt matter dim wise
# # support_data.view(-1,C,T,H,W)
# A, B, C, D, E = model(support_data,query_data,supp_cls,support_proto,support_trimmed,mode="train")

# print(A.size())


# ############### End #################




# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils.transformer import SnippetEmbedding
# import yaml
# from scipy import ndimage
# import itertools,operator
# # from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
# from config.dataset_class import activity_dict
# from config.zero_shot import split_t1_train, split_t1_test, split_t2_train, split_t2_test , t1_dict_train , t1_dict_test , t2_dict_train , t2_dict_test
# from config.few_shot import base_class,val_class,test_class,base_dict,val_dict,test_dict, base_train,base_train_dict
# # from denseclip.untils import tokenize
# # from denseclip.models import Transformer,LayerNorm
# # import denseclip
# from MaskFormer.mask_former.modeling.transformer.transformer_predictor_v2 import TransformerPredictor
# from transformers import CLIPTokenizer, CLIPModel
# from transformers import CLIPTextModel, CLIPTextConfig
# import matplotlib.patheffects as PathEffects
# import seaborn as sns
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})

# from utils.backbone_utils import prepare_backbone
# from AdaptFormer.models.vit_video_v2 import VisionTransformer
# from functools import partial
# from easydict import EasyDict
# # from config.dataset_class import activity_dict, thumos_dict , thumos_dict2



# with open("./config/anet_2gpu.yaml", 'r', encoding='utf-8') as f:
#         tmp = f.read()
#         config = yaml.load(tmp, Loader=yaml.FullLoader)

# tun_config = EasyDict(
#         # AdaptFormer
#         ffn_adapt=True,
#         ffn_option="parallel",
#         ffn_adapter_layernorm_option="none",
#         ffn_adapter_init_option="lora",
#         ffn_adapter_scalar="0.1",
#         ffn_num=64,
#         d_model=config['pretraining']['emb_dim'],
#         # VPT related
#         vpt_on=False,
#         vpt_num=1,
#     )
    

# class STALE(nn.Module):
#     def __init__(self):
#         super(STALE, self).__init__()
#         self.len_feat = config['model']['feat_dim']
#         self.temporal_scale = config['model']['temporal_scale'] 
#         self.split = config['dataset']['split']
#         self.num_classes = config['dataset']['num_classes']+1
#         self.n_heads = config['model']['embedding_head']
#         self.clip_stride = config['pretraining']['clip_stride']
#         self.clip_max = config['pretraining']['clip_length']
#         self.emb_dim = config['pretraining']['emb_dim']
#         self.num_segment = int(self.clip_max / self.clip_stride)
#         self.embedding = SnippetEmbedding(self.n_heads, self.emb_dim, self.emb_dim, self.emb_dim, dropout=0.2)
#         self.cross_att = SnippetEmbedding(self.n_heads, self.emb_dim, self.emb_dim, self.emb_dim, dq_model=self.len_feat, dropout=0.3)
#         self.query_adapt = SnippetEmbedding(self.n_heads, self.emb_dim, self.emb_dim, self.emb_dim, dropout=0.3)
#         self.context_length = 30
#         self.txt_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#         self.nshot = config['fewshot']['shot']
#         self.cl_names = list(activity_dict.keys())
#         self.contxt_token = config['fewshot']['num_context']
#         self.supp_trim = config['fewshot']['trim_support']
#         self.delta = 0
#         self.num_queries = 1
#         self.bg_embeddings = nn.Parameter(
#             torch.empty(1, 512)
#         )

#         self.proj = nn.Sequential(
#             nn.Conv1d(self.emb_dim, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#         self.proj_txt = nn.Sequential(
#             nn.Conv1d(512,self.emb_dim, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#         context_length = self.context_length
#         self.contexts = nn.Parameter(torch.randn(1, context_length))
#         nn.init.trunc_normal_(self.contexts)

#         ##### ViT-B/16 Transformer Backbone ######
#         self.video_backbone = VisionTransformer(
#             # patch_size=16, 
#             # embed_dim=768, 
#             # depth=12, 
#             # num_heads=12, 
#             # mlp_ratio=4, 
#             # qkv_bias=True,
#             # norm_layer=partial(nn.LayerNorm, eps=1e-6),
#             img_size=112, 
#             patch_size=16, 
#             in_chans=3, 
#             num_classes=1000, 
#             embed_dim=self.emb_dim, # 768
#             depth=1, # 12
#             num_heads=4, 
#             mlp_ratio=4., 
#             qkv_bias=False, 
#             qk_scale=None, 
#             drop_rate=0., 
#             attn_drop_rate=0.,
#             drop_path_rate=0., 
#             norm_layer=partial(nn.LayerNorm, eps=1e-6), 
#             init_values=0.,
#             use_learnable_pos_emb=False, 
#             init_scale=0.,
#             all_frames=self.clip_stride,
#             tubelet_size=2,
#             use_mean_pooling=True,
#             tuning_config=tun_config
#         ).cuda()

#         # self.video_backbone = torch.nn.DataParallel(self.video_backbone, device_ids=[0]).cuda()
#         self.backbone_ckpt = torch.load(config['pretraining']['video_transformer'], map_location='cpu')

#         # self.video_backbone.load_state_dict()
#         self.backbone_encoder = prepare_backbone(self.video_backbone,self.backbone_ckpt)

#         self.masktrans = TransformerPredictor(
#             in_channels=768,
#             mask_classification=False,
#             num_classes=self.num_classes,
#             hidden_dim=self.emb_dim,
#             num_queries=1,
#             nheads=2,
#             dropout=0.1,
#             dim_feedforward=1,
#             enc_layers=2,
#             dec_layers=2,
#             pre_norm=True,
#             deep_supervision=False,
#             mask_dim=self.emb_dim,
#             enforce_input_project=True
#         ).cuda()


#         self.classifier = nn.Sequential(
#             nn.Conv1d(in_channels=self.len_feat, out_channels=self.num_classes+1, kernel_size=1,
#             padding=0)
#         )
    
#         self.cls_adapter = nn.Sequential(
#             nn.Conv1d(in_channels=self.len_feat+self.emb_dim, out_channels=self.emb_dim, kernel_size=1,
#             padding=0)
#         )

#         self.bn1 = nn.BatchNorm1d(num_features=2048)
#         self.localizer_mask = nn.Sequential(
#             nn.Conv1d(in_channels=self.emb_dim, out_channels=256, kernel_size=3,padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=256, out_channels=self.temporal_scale, kernel_size=1,stride=1, padding=0, bias=False),
#             nn.Sigmoid()
#         )

#         self.avg_pool = torch.nn.AdaptiveAvgPool1d(self.contxt_token)

#         self.reduce_mask = nn.Sequential(
#             nn.Conv1d(100, 1, 1),
#             nn.Sigmoid()
#         )

#         self.mask_MLP = nn.Sequential(
#             nn.Conv1d(5,1,1),
#             nn.Sigmoid()
#         )

#     def get_prompt(self,cl_names,cl_dict):
#         temp_prompt = []
#         label_id = []
#         meta_learn_class = config['fewshot']['meta_class']
#         for c in cl_names:
#             if meta_learn_class:
#                 temp_prompt.append(c)
#                 label_id.append(cl_dict[c])
#             else:
#                 temp_prompt.append("a video of action"+" "+c)
#                 label_id.append(cl_dict[c])

#         return temp_prompt, label_id

#     def projection(self,loc_feat):
#         proj_feat = self.proj(loc_feat)
#         return proj_feat

#     def compute_score_maps(self, visual, text):

#         B,K,C = text.size()
#         text_cls = text[:,:(K-1),:]
#         text_cls = text_cls / text_cls.norm(dim=2, keepdim=True)
#         text = text / text.norm(dim=2, keepdim=True)
#         visual = torch.clamp(visual,min=1e-4)
#         visual_cls = visual.mean(dim=2)
#         visual = visual / visual.norm(dim=1, keepdim=True)
#         visual_cls = visual_cls / visual_cls.norm(dim=1, keepdim=True)
#         score_cls = torch.einsum('bc,bkc->bk', visual_cls, text_cls) * 100
#         score_map = torch.einsum('bct,bkc->bkt', visual, text) * 100

#         return score_map, score_cls

    

#     def crop_features(self, feature, mask):
#         # print(feature.size(),mask.size())
#         dtype = mask.dtype
#         trim_ten = []
#         trim_feat = torch.zeros_like(feature)
#         mask_fg = torch.ones_like(mask)
#         mask_bg = torch.zeros_like(mask)
#         for i in range(mask.size(0)):
#             cls_thres = float(torch.mean(mask[i,:],dim=0).detach().cpu().numpy())
#             top_mask = torch.where(mask[i,:] >= cls_thres, mask_fg[i,:], mask_bg[i,:]).cuda(0)
#             top_loc = (top_mask==1).nonzero().squeeze().cuda(0)
#             trim_feat[i,:,top_loc] = feature[i,:,top_loc]
#             trim_ten.append(trim_feat)
#         if len(trim_ten) == 0:
#             trim_ten = feature
#         else:
#             trim_ten = torch.stack(trim_ten, dim=0)
    
#         return trim_feat


#     def save_plots(self,embedding,test_dict):

#         key_list = list(test_dict.keys())
#         val_list = list(test_dict.values())

#         _, label_list = self.get_prompt(test_dict.keys(),test_dict)

#         text_cls = embedding.detach().cpu().numpy()
#         fashion_tsne = TSNE(random_state=123).fit_transform(text_cls)

#         print('t-SNE done!')

#         x = fashion_tsne 
#         colors = np.asarray(label_list)
#         num_classes = len(np.unique(colors))
#         palette = np.array(sns.color_palette("hls", num_classes))

#         # create a scatter plot.
#         plt.figure(figsize=(16, 16))
#         ax = plt.subplot(aspect='equal')
#         sc = ax.scatter(x[:,0], x[:,1], lw=20, s=60, c=palette[colors.astype(np.int)])
#         plt.xlim(-25, 25)
#         plt.ylim(-25, 25)
#         ax.axis('off')
#         ax.axis('tight')


#         # add the labels for each digit corresponding to the label
#         txts = []

#         for i in range(num_classes):

#             # Position of each label at median of data points.

#             xtext, ytext = np.median(x[colors == i, :], axis=0)
#             txt = ax.text(xtext, ytext, key_list[val_list.index(i)] , fontsize=13)
#             txt.set_path_effects([
#                 PathEffects.Stroke(linewidth=5, foreground="w"),
#                 PathEffects.Normal()])
#             txts.append(txt)
            
#         plt.savefig('tsne.png')


#     def gen_text_query(self,vid_feat,sub_cls,sub_proto, mode):
#         B,T,C = vid_feat.size()
#         if len(sub_cls) == 0:
#             if self.nshot == 0:
#                 if mode == "train" and self.split == 50:
#                     cl_names = list(t2_dict_train.keys())
#                     self.num_classes = 100
#                 elif mode == "test" and self.split == 50:
#                     cl_names = list(t2_dict_test.keys())
#                     self.num_classes = 100
#                 elif mode == "train" and self.split == 75:
#                     cl_names = list(t1_dict_train.keys())
#                     self.num_classes = 150
#                 elif mode == "test" and self.split == 75:
#                     cl_names = list(t1_dict_test.keys())
#                     self.num_classes = 50
#             else: 
#                 if mode == "train":
#                     cl_names = list(base_train_dict.keys())
#                     self.num_classes = 180
#                 elif mode == "test":
#                     cl_names = list(test_dict.keys())
#                     self.num_classes = 20
#         else:
#             cl_names = sub_cls
#             self.num_classes = len(sub_cls)

#         act_prompt,_ = self.get_prompt(cl_names,test_dict)
#         texts = self.tokenizer(act_prompt, padding=True, return_tensors="pt").to('cuda') 
#         text_cls = self.txt_model.get_text_features(**texts) ## [cls,txt_feat] --> [200,512]
#         text_emb = torch.mean(text_cls,dim=0)

#         return text_emb
        
#     def text_features(self,vid_feat, sub_cls, sup_proto, mode):
#         B,T,C = vid_feat.size()
#         if len(sub_cls) == 0:
#             if self.nshot == 0:
#                 if mode == "train" and self.split == 50:
#                     cl_names = list(t2_dict_train.keys())
#                     self.num_classes = 100
#                 elif mode == "test" and self.split == 50:
#                     cl_names = list(t2_dict_test.keys())
#                     self.num_classes = 100
#                 elif mode == "train" and self.split == 75:
#                     cl_names = list(t1_dict_train.keys())
#                     self.num_classes = 150
#                 elif mode == "test" and self.split == 75:
#                     cl_names = list(t1_dict_test.keys())
#                     self.num_classes = 50
#             else: 
#                 if mode == "train":
#                     cl_names = list(base_train_dict.keys())
#                     self.num_classes = 180
#                 elif mode == "test":
#                     cl_names = list(test_dict.keys())
#                     self.num_classes = 20
#         else:
#             cl_names = sub_cls
#             self.num_classes = len(sub_cls)
        
#         sub_cls_dict = {}
#         nway = self.num_classes
#         nshot = int(B/nway)
#         act_prompt,_ = self.get_prompt(cl_names,test_dict)
#         meta_learn_class = config['fewshot']['meta_class']


#         if meta_learn_class :

#             texts = self.tokenizer(act_prompt, padding=True, return_tensors="pt").to('cuda')
#             cls_text_emb = texts["input_ids"]
#             context = sup_proto.squeeze(1).type(torch.LongTensor).to('cuda')
#             text_context = torch.cat([context,cls_text_emb],dim=1)
#             text_cls = self.txt_model.get_text_features(text_context) ## [cls,txt_feat] --> [200,512]
#             text_emb = torch.cat([text_cls,self.bg_embeddings],dim=0).expand(B,-1,-1)  ## [bs, cls+1 ,txt_feat] --> [bs,201,512]
#         else:
#             texts = self.tokenizer(act_prompt, padding=True, return_tensors="pt").to('cuda')
#             text_cls = self.txt_model.get_text_features(**texts) ## [cls,txt_feat] --> [200,512]
#             text_emb = torch.cat([text_cls,self.bg_embeddings],dim=0).expand(B,-1,-1)  ## [bs, cls+1 ,txt_feat] --> [bs,201,512]

#         return text_emb
    

#     def find_mask(self,raw_mask):
#         seq = raw_mask.detach().cpu().numpy()
#         # print(seq)
#         m_th = np.mean(seq)
#         filtered_seq = seq > 0.5
#         integer_map = map(int,filtered_seq)
#         filtered_seq_int = list(integer_map)
#         filtered_seq_int2 = ndimage.binary_fill_holes(filtered_seq_int).astype(int).tolist()
#         if 1 in filtered_seq_int2 :
#             r = max((list(y) for (x,y) in itertools.groupby((enumerate(filtered_seq_int2)),operator.itemgetter(1)) if x == 1), key=len)
#             if r[-1][0] - r[0][0] > 1:
#                 start_pt = r[0][0]
#                 end_pt = r[-1][0]
#             else:
#                 start_pt = 0
#                 end_pt = 99
#         else:
#             start_pt = 0
#             end_pt = 99
#         # print(start_pt,end_pt)
#         return start_pt,end_pt


#     def forward(self, sup_snip, query_snip, sub_cls, sup_proto_gt, sup_trim, mode):

#         raw_vid = sup_snip
#         snip_feat = [] 
#         for i in range(self.num_segment):
#             win_feat = self.backbone_encoder(raw_vid[:,:,self.clip_stride*i:self.clip_stride*i+self.clip_stride,:,:]) ## ip : [batch,channel=3,stride=16,H,W] op: [batch,emb_dim]
#             snip_feat.append(win_feat)

#         snip =  torch.cat(snip_feat,1).view(-1,self.num_segment,self.emb_dim) ## [batch,num_segment,emb_dim]
#         snip = snip.permute(0,2,1)
#         sup_snip = F.interpolate(snip, size=self.temporal_scale, mode='linear',align_corners=False) ### interpolate temporal dimension to 100 ## [batch,100,emb_dim]
#         vid_feature = sup_snip

#         if mode == "test":
#             raw_vid_alt = query_snip
#             snip_feat_q = [] 
#             for i in range(self.num_segment):
#                 win_feat = self.backbone_encoder(raw_vid_alt[:,:,self.clip_stride*i:self.clip_stride*i+self.clip_stride,:,:]) ## ip : [batch,channel=3,stride=16,H,W] op: [batch,emb_dim]
#                 snip_feat_q.append(win_feat)

#             snip_alt =  torch.cat(snip_feat_q,1).view(-1,self.num_segment,self.emb_dim) ## [batch,num_segment,emb_dim]
#             snip_alt = snip_alt.permute(0,2,1)
#             q_snip = F.interpolate(snip_alt, size=self.temporal_scale, mode='linear',align_corners=False) ### interpolate temporal dimension to 100 ## [batch,100,emb_dim]
#             vid_feature_alt = q_snip



#         # print(mode,)
#         B,C,T = vid_feature.size()

#         snip = sup_snip.permute(0,2,1)
#         out = self.embedding(snip,snip,snip)
#         out = out.permute(0,2,1)
#         features = out

#         if mode == "train":
#             nway,nshot,feat_dim,temp_dim = sup_proto_gt.size()
#             sup_proto_trimmed = vid_feature.view(nway,nshot,feat_dim,temp_dim)*sup_proto_gt
#             sup_proto = torch.mean(sup_proto_trimmed,dim=1) ## nshot consumed
#             sup_proto = torch.mean(sup_proto, dim = 2) ## tmp_dim consumed
#         else:

#             print(' -- debug', sup_proto_gt.shape)
#             nway,nshot,feat_dim,temp_dim = sup_proto_gt.size()
#             print(' -- debug', vid_feature_alt.shape)
#             sup_proto_trimmed = vid_feature_alt.view(nway,nshot,feat_dim,temp_dim)*sup_proto_gt
#             sup_proto = torch.mean(sup_proto_trimmed,dim=1) ## nshot consumed
#             sup_proto = torch.mean(sup_proto, dim = 2) ## tmp_dim consumed
        
            
#        ##### Extract the video features using Adaptformer #######
# #         raw_vid = sup_snip
# #         snip_feat = [] 
# #         for i in range(self.num_segment):
# #             win_feat = self.backbone_encoder(raw_vid[:,:,self.clip_stride*i:self.clip_stride*i+self.clip_stride,:,:]) ## ip : [batch,channel=3,stride=16,H,W] op: [batch,emb_dim]
# #             snip_feat.append(win_feat)

# #         snip =  torch.cat(snip_feat,1).view(-1,self.num_segment,self.emb_dim) ## [batch,num_segment,emb_dim]
# #         snip = snip.permute(0,2,1)
# #         sup_snip = F.interpolate(snip, size=self.temporal_scale, mode='linear',align_corners=False) ### interpolate temporal dimension to 100 ## [batch,100,emb_dim]


# #         vid_feature = sup_snip
        
# #         # print(mode,)
# #         B,C,T = vid_feature.size()

# #         snip = sup_snip.permute(0,2,1)
# #         out = self.embedding(snip,snip,snip)
# #         out = out.permute(0,2,1)
# #         features = out

#         #### Vision-Language fusion query #### 
#         fg_text_emb = self.gen_text_query(features,sub_cls,sup_proto,mode) ###[1, dim]
#         # print("text",fg_text_emb.size())
#         fg_vid_emb = torch.mean(torch.mean(features,dim=2),dim=0) ### [1,dim]
#         # print("vid", fg_vid_emb.size())
#         fuse_emb = torch.cat([fg_vid_emb,fg_text_emb],dim=0).unsqueeze(0).unsqueeze(2)
#         # print(fuse_emb.size())
#         fg_query = self.cls_adapter(fuse_emb).squeeze(2).expand(self.num_queries,-1)

#         ### Action Mask Localizer Branch ###
#         if mode == "train":
#             bottom_br = self.localizer_mask(features)
#         else:
#             bottom_br = self.localizer_mask(features)

#         #### Few-Shot Specific #####
#         sup_context = self.avg_pool(sup_proto.unsqueeze(1))

#         #### Representation Mask ####
#         snipmask = self.masktrans(vid_feature.unsqueeze(2),features.unsqueeze(3),fg_query)
#         bot_mask = torch.mean(bottom_br, dim=2)
#         soft_mask = torch.sigmoid(snipmask["pred_masks"]).view(-1,self.temporal_scale)
#         mask_feat = self.crop_features(features,soft_mask)
#         soft_tensor = bot_mask
#         text_feat = self.text_features(features, sub_cls, sup_context, mode)
#         text_feat = self.proj_txt(text_feat.permute(0,2,1))
#         text_feat = text_feat.permute(0,2,1)
#         mask_feat = mask_feat.permute(0,2,1)

#         #### Vision-Language Cross-Adaptation ####
#         text_feat_att = self.cross_att(text_feat, mask_feat, mask_feat)
#         text_feat_fin = text_feat_att + text_feat
#         mask_feat = mask_feat.permute(0,2,1)
#         score_maps, score_maps_class = self.compute_score_maps(mask_feat, text_feat_fin)
        
#         #### Contextualized Vision-Language Classifier #####
#         top_br = score_maps
#         print(top_br)
#         query_adapt_feat = self.query_adapt(features.permute(0,2,1), mask_feat.permute(0,2,1),mask_feat.permute(0,2,1))
#         bottom_br = self.localizer_mask(query_adapt_feat.permute(0,2,1))
#         # print(mode, top_br.size(), bottom_br.size() , soft_tensor.size() , score_maps_class.size(), features.size())

#         return top_br, bottom_br , soft_tensor , score_maps_class, features



#     def extract_feature(self, snip):

#         vid_feature = snip
#         snip = snip.permute(0,2,1)
#         out = self.embedding(snip,snip,snip)
#         out = out.permute(0,2,1)
#         features = out

#         snipmask = self.masktrans(vid_feature.unsqueeze(2),features.unsqueeze(3))

#         ### Global Segmentation Mask Branch ###
#         bottom_br = self.global_mask(features)

#         bot_mask = torch.mean(bottom_br, dim=1)


#         soft_mask = torch.sigmoid(snipmask["pred_masks"]).view(-1,self.temporal_scale)


#         # mask_feat = self.crop_features(features,soft_mask)
#         mask_feat = self.crop_features(features,soft_mask)

#         # soft_tensor = bot_mask
#         soft_tensor = soft_mask

#         text_feat = self.text_features(vid_feature, mode)
#         # print(mask_feat)

#         return mask_feat , text_feat








