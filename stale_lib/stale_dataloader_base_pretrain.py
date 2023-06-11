#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import torch.utils.data as data
import torch
import h5py
from torch.functional import F
import os
import math
from config.dataset_class import activity_dict
import yaml
import tqdm
from config.few_shot import base_class,val_class,test_class,base_dict,val_dict,test_dict, base_train,base_train_dict


with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class STALEDataset(data.Dataset):
    def __init__(self, subset="train", mode="train"):
        self.temporal_scale = config['model']['temporal_scale']
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = mode
        self.feature_path = config['training']['feature_path']
        self.video_info_path = config['dataset']['training']['video_info_path']
        self.video_anno_path = config['dataset']['training']['video_anno_path']
        self.num_frame = config['dataset']['training']['num_frame']
        self.num_classes = config['fewshot']['num_base']
        self.class_to_idx = base_train_dict
        video_infos = self.get_video_info(self.video_info_path)
        self.info = video_infos
        video_annos = self.get_video_anno(video_infos, self.video_anno_path)
        self.subset_mask = self.getVideoMask(video_annos,self.temporal_scale)
        self.subset_mask_list = list(self.subset_mask.keys())
        self.ismulti = config['fewshot']['ismulti']
        self.vid_path = config['pretraining']['video_path']


        
    def get_video_anno(self,video_infos,video_anno_path):

        anno_database = load_json(self.video_anno_path)
        # print(anno_database)
        exclude_videos = [
            "v_s82_J03bqwQ",
            "v_H-5nHSHwFOk",
            "v_YtgiDWEY_1A",
            "v_ndET50Ccnr8",
            "v_gk6NAPqfJoY",
            "v_VdeYnCIbRJ4",
            "v_MdrK2uQ-GvA"
        ]
        video_dict = {}
        for video_name in video_infos.keys():
            if video_name in exclude_videos:
                print('Exclude video',video_name )
                continue
            video_info = anno_database[video_name]
            video_subset = video_infos[video_name]['subset']
            if self.subset in video_subset:
                video_info.update({'subset': video_subset})
                video_dict[video_name] = video_info


        return video_dict

    

    def get_video_info(self,video_info_path):

        df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
        video_infos = {}
        self.v_list ={}
        for info in df_info:
            video_infos[info[0]] = {
                'duration': info[2],
                'subset': info[5]
            }
            
        return video_infos

    def getAnnotation(self, subset, anno):
        # print()

        if subset == "train":
            cls_dict = base_train_dict
            self.class_to_idx = base_train_dict
        elif subset == "validation":
            cls_dict = test_dict
            self.class_to_idx = test_dict
        temporal_dict={}
        for idx in anno.keys():
            labels = anno[idx]['annotations']
            subset_vid  = anno[idx]['subset']
            num_frame = anno[idx]['feature_frame']
            vid_frame = anno[idx]['duration_frame']
            num_sec = anno[idx]['duration_second']
            corr_sec = float(num_frame) / vid_frame * num_sec
            label_list= []
            if subset in subset_vid:
                for j in range(len(labels)):
                    tmp_info = labels[j]
                    clip_factor = self.temporal_scale / (corr_sec * (self.num_frame+1))
                    action_start = tmp_info['segment'][0]*clip_factor
                    snip_start = max(min(1, tmp_info['segment'][0] / corr_sec), 0)
                    action_end = tmp_info['segment'][1]*clip_factor
                    snip_end = max(min(1, tmp_info['segment'][1] / corr_sec), 0)
                    gt_label = tmp_info["label"]
                    # print(gt_label, list(base_train_dict.keys()))
                if action_end - action_start > 1 and gt_label in list(cls_dict.keys()):
                    label_list.append([snip_start,snip_end,gt_label])    
            if len(label_list)>0:
                temporal_dict[idx]= {"labels":label_list,
                                    "video_duration": num_sec}

        return temporal_dict

    def getVideoMask(self,video_annos,clip_length=100):

        self.video_mask = {}
        idx_list = self.getAnnotation(self.subset,video_annos)
        # print(len(list(idx_list.keys())))
        self.anno_final = idx_list
        self.anno_final_idx = list(idx_list.keys())
        print('Loading '+self.subset+' Video Information (Base Class)...')
        for idx in tqdm.tqdm(list(idx_list.keys()),ncols=0):
            # print(os.path.exists(os.path.join(self.feature_path+"/",idx+".npy")))
            feature_path = os.path.join(self.feature_path,idx+".npy")
            # if os.path.exists(feature_path)and idx in list(idx_list.keys()):
            if idx in list(idx_list.keys()):
                # print(idx)
                cur_annos = idx_list[idx]["labels"]
                mask_list=[]
                for l_id in range(len(cur_annos)):
                    mask_start = int(math.floor(clip_length*cur_annos[l_id][0]))
                    mask_end = int(math.floor(clip_length*cur_annos[l_id][1]))
                    mask_label_idx = self.class_to_idx[cur_annos[l_id][2]]
                    mask_list.append([mask_start,mask_end,mask_label_idx])
                self.video_mask[idx] = mask_list
            
            else:
                print('File not found:', feature_path)    

        return self.video_mask
                
    def loadFeature(self, idx):
        feat = np.load(os.path.join(self.feature_path, idx+".npy"))
        feat_tensor = torch.Tensor(feat)
        video_data = torch.transpose(feat_tensor, 0, 1)
        video_data = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
       
        return video_data

    ##### Loader for Raw Video Frames #####
    def loadVideo(self, idx):

        frames = np.load(os.path.join(self.vid_path, idx+".npy"),allow_pickle=True)
        # frames = np.random.rand(768,224,224,3)
        frames = np.transpose(frames, [3, 0, 1, 2]).astype(np.float) ### (C,T,H.,W)
        c, t, h, w = frames.shape
        if t < config["pretraining"]["clip_length"]:
            pad_t = config["pretraining"]["clip_length"] - t
            zero_clip = np.ones([c, pad_t, h, w], dtype=frames.dtype) * 127.5
            frames = np.concatenate([frames, zero_clip], 1)

        input_data = torch.from_numpy(frames.copy()).float()
        vid_tensor = torch.Tensor(input_data)

        return vid_tensor


    def getVideoData(self,index):

        if self.subset == 'validation':
            self.num_classes = 200 - config['fewshot']['num_base']

        mask_idx = self.subset_mask_list[index]
        mask_data = self.loadVideo(mask_idx)
        mask_label = self.video_mask[mask_idx]
        bbox = np.array(mask_label)
        start_id = bbox[:,0]
        end_id = bbox[:,1]
        label_id = bbox[:,2]
        cls_mask = np.zeros([self.num_classes+1, self.temporal_scale]) ## dim : 201x100
        temporary_mask = np.zeros([self.temporal_scale])
        action_mask = np.zeros([self.temporal_scale,self.temporal_scale]) ## dim : 100 x 100
        cas_mask = np.zeros([self.num_classes])
    
        start_indexes = []
        end_indexes = []
        tuple_list =[]
        
        for idx in range(len(start_id)):
          lbl_id = label_id[idx]
          start_indexes.append(start_id[idx]+1)
          end_indexes.append(end_id[idx]-1)
          tuple_list.append([start_id[idx]+1, end_id[idx]-1,lbl_id])
        temp_mask_cls = np.zeros([self.temporal_scale])

        for idx in range(len(start_id)):
            temp_mask_cls[tuple_list[idx][0]:tuple_list[idx][1]]=1
            lbl_idx = int(tuple_list[idx][2])
            cls_mask[lbl_idx,:]= temp_mask_cls
            
        for idx in range(len(start_id)):
          temporary_mask[tuple_list[idx][0]:tuple_list[idx][1]] = 1
 
        background_mask = 1 - temporary_mask
        v_label = np.zeros([1])
        new_mask = np.zeros([self.temporal_scale])
        for p in range(self.temporal_scale):
            new_mask[p] = -1 

        cls_mask[self.num_classes,:] = background_mask
        filter_lab = list(set(label_id))
        for j in range(len(filter_lab)):
            label_idx = filter_lab[j]
            cas_mask[label_idx] = 1

        for idx in range(len(start_indexes)):
          len_gt = int(end_indexes[idx] - start_indexes[idx])
          mod_start = tuple_list[idx][0]
          mod_end = tuple_list[idx][1]
          new_lab = tuple_list[idx][2]
          new_mask[mod_start:mod_end] = new_lab

        for p in range(self.temporal_scale):
            if new_mask[p] == -1:
                new_mask[p] = self.num_classes

        classifier_branch = torch.Tensor(new_mask).type(torch.LongTensor)

        for idx in range(len(start_indexes)):
            len_gt = int(end_indexes[idx] - start_indexes[idx])
            mod_start = tuple_list[idx][0]
            mod_end = tuple_list[idx][1]
            action_mask[mod_start:(mod_end), mod_start:(mod_end)] = 1

        global_mask_branch = torch.Tensor(action_mask)
        cas_mask = torch.Tensor(cas_mask)
        mask_top = torch.Tensor(cls_mask)
        v_label = torch.Tensor()
        fg_mask = torch.Tensor(temporary_mask)
        bg_mask = torch.Tensor(background_mask)
        bot_mask = fg_mask
        return mask_data, classifier_branch,global_mask_branch,mask_top,cas_mask,bot_mask


    def __getitem__(self, index):
        
        mask_data, top_branch, bottom_branch, mask_top, cas_mask , bot_mask = self.getVideoData(index)
        if self.mode == "train":
            return mask_data,top_branch,bottom_branch,mask_top,cas_mask, bot_mask
        else:
            return index, mask_data






    def __len__(self):
        return len(self.subset_mask_list)
