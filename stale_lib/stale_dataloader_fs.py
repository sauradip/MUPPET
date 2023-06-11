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
from config.zero_shot import split_t1_train, split_t1_test, split_t2_train, split_t2_test , t1_dict_train , t1_dict_test , t2_dict_train , t2_dict_test
from config.few_shot import base_class,val_class,test_class,base_dict,val_dict,test_dict
import yaml
import tqdm
import random



with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class STALEEpisodicDataset(data.Dataset):
    def __init__(self, subset="train", mode="train"):
        # super().__init__()
        self.temporal_scale = config['model']['temporal_scale']
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = mode
        self.zs = True
        self.feature_path = config['training']['feature_path']
        self.video_info_path = config['dataset']['training']['video_info_path']
        self.video_anno_path = config['dataset']['training']['video_anno_path']
        self.num_frame = config['dataset']['training']['num_frame']
        self.split = config['dataset']['split']
        self.num_classes = config['fewshot']['num_way']
        self.class_to_idx = test_dict
        video_infos = self.get_video_info(self.video_info_path)
        self.info = video_infos
        video_annos = self.get_video_anno(video_infos, self.video_anno_path)
        self.annos = video_annos
        self.subset_mask = self.getVideoMask(video_annos,self.temporal_scale)
        ###### few shot params ####
        self.fsmode = config['fewshot']['mode']
        self.fshot = config['fewshot']['shot'] 
        self.istrimmed = config['fewshot']['trimmed']
        self.anno_db_cwise,self.annos_label = self.getAnnotationcwise(self.subset, video_annos)
        self.fway = config['fewshot']['num_way']
        # print(self.subset_mask.keys())
        self.subset_mask_list = list(self.annos_label.keys())
        self.is_trim = config['fewshot']['trimmed']

        
        
        

    def get_video_annocwise(self,video_infos,video_anno_path):

        anno_database = load_json(self.video_anno_path)
        # print(anno_database)
        video_dict = {}
        for video_name in video_infos.keys():
            video_info = anno_database[video_name]
            video_subset = video_infos[video_name]['subset']
            if self.subset in video_subset:
                video_info.update({'subset': video_subset})
                video_dict[video_name] = video_info

        return video_dict


    def get_video_infocwise(self,video_info_path):

        df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
        video_infos = {}
        self.v_list ={}
        for info in df_info:
            video_infos[info[0]] = {
                'duration': info[2],
                'subset': info[5]
            }
            
        return video_infos

    


    def getAnnotationcwise(self, subset, anno):
        temporal_dict={}
        annts_cwise = {}
        cnt=0
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
                    clip_factor = self.temporal_scale / ( corr_sec * (self.num_frame+1) )
                    action_start = tmp_info['segment'][0]*clip_factor
                    snip_start = max(min(1, tmp_info['segment'][0] / corr_sec), 0)
                    action_end = tmp_info['segment'][1]*clip_factor
                    snip_end = max(min(1, tmp_info['segment'][1] / corr_sec), 0)
                    gt_label = tmp_info["label"]
                    # print(gt_label)

                if self.fsmode == 1 : 
                    if action_end - action_start > 1 :
                        label_list.append([snip_start,snip_end,gt_label])
                        cnt+=1
                        if gt_label not in annts_cwise:
                            annts_cwise[gt_label] = []
                        annts_cwise[gt_label].append(idx) ##just index

                elif self.fsmode == 2 or self.fsmode == 3 :  ### only novel-support or query class
                    if action_end - action_start > 1 and gt_label in test_class:
                        # print(gt_label)
                        label_list.append([snip_start,snip_end,gt_label])
                        cnt+=1
                        if gt_label not in annts_cwise:
                            annts_cwise[gt_label] = []
                        annts_cwise[gt_label].append(idx) ##just index
            if len(label_list)>0:
                temporal_dict[idx]= {"labels":label_list,
                                    "video_duration": num_sec}

        return annts_cwise, temporal_dict
        # return annts_cwise



    def getVideoMaskcwise(self,video_annos,idx,sub_cls_dict,clip_length=100):

        if os.path.exists(os.path.join(self.feature_path+"/",idx+".npy")) and idx in list(video_annos.keys()):
                cur_annos = video_annos[idx]["labels"]
                mask_list=[]
                for l_id in range(len(cur_annos)):
                    mask_start = int(math.floor(clip_length*cur_annos[l_id][0]))
                    mask_end = int(math.floor(clip_length*cur_annos[l_id][1]))
                    if cur_annos[l_id][2] in sub_cls_dict.keys():
                        mask_label_idx = sub_cls_dict[cur_annos[l_id][2]]
                        mask_list.append([mask_start,mask_end,mask_label_idx])
        else:#
            print("File not found, look for ", os.path.join(self.feature_path+"/",idx+".npy"))

        return mask_list



    def loadFeaturecwise(self, video_name):

        feat_path = self.feature_path
        feat = np.load(os.path.join(feat_path, video_name+".npy"))
        feat_tensor = torch.Tensor(feat)
        video_data = torch.transpose(feat_tensor, 0, 1)
        video_data = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
       
        return video_data



    def getBranchData(self, mask_label):

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

        fg_mask = torch.Tensor(temporary_mask)
        bg_mask = torch.Tensor(background_mask)
        bot_mask = fg_mask
        v_label = torch.Tensor()

        return classifier_branch,global_mask_branch,mask_top,cas_mask,bot_mask,start_id,end_id


    def getEpisodicVideoData(self,index):
    
        nshot = self.fshot ### number of examples per class

        nway = self.fway ### number of class per support set

        class_chosen = []
        rand_cls_list = []

        cls_support_video = []
        cls_support_video_trim = []
        cls_support_label_mask = []
        cls_support_label_loc = []
        cls_support_label_class = []
        cls_support_label_cgt = []
        cls_support_label_cagt = []
        cls_support_label_mgt = []

        cls_query_video = []
        cls_query_video_trim = []
        cls_query_label_mask = []
        cls_query_label_loc = []
        cls_query_label_class = []
        cls_query_label_cgt = []
        cls_query_label_cagt = []
        cls_query_label_mgt = []

        subcls_list = []
        query_vid_list = []
        support_vid_list = []
        sub_cls_dict = {}
        # class_list = []

         # ========= Read Query Video + Choose Label ======= #

        cwise_anno = self.anno_db_cwise #### to change ## classwise video list
        nu = 0
        tst_cl_list = test_class
        for cl in range(nway):
            ran_cls = random.choice(tst_cl_list)
            rand_cls_list.append(ran_cls) ## query label from test_Class
            tst_cl_list.remove(ran_cls)
            sub_cls_dict[ran_cls] = nu
            nu+=1
        
        for class_chosen in rand_cls_list:

            support_video_list = []
            support_video_list_trim = []
            support_label_list_mask = []
            support_label_list_loc = []
            support_label_list_class = []
            support_label_list_cgt = []
            support_label_list_cagt = []
            support_label_list_mgt = []
            subcls_list.append(list(self.class_to_idx.keys()).index(class_chosen)) ## index of chosen class among novel class
            rand_vid_name = random.choice(cwise_anno[class_chosen]) ### get random video for each support class
            query_vid_list.append(rand_vid_name)
            query_label = self.getVideoMaskcwise(self.annos_label, rand_vid_name, sub_cls_dict) ## list of [start,end,label] per video
            query_label_class, query_label_loc, query_label_mask, q_lb_cgt, q_lb_mgt , strt_idx, end_idx = self.getBranchData(query_label) ## gt fpormation from label
            query_data = self.loadFeaturecwise(rand_vid_name) ## backbone features
            query_data_zero = torch.zeros_like(query_data)

            if self.is_trim:
                for i in range(len(strt_idx)):
                    query_data_zero[:,strt_idx[i]+1:end_idx[i]-1] = query_data[:,strt_idx[i]+1:end_idx[i]-1]
                query_mask = query_data_zero !=0
                query_data_trimmed = (query_data_zero*query_mask).sum(dim=1)/query_mask.sum(dim=1)
                query_data_trimmed = query_data_trimmed.unsqueeze(1).expand(-1,query_data.size(1))
            else:
                query_data_trimmed = query_data_zero

            C,T = query_data.size() 
            file_class_chosen = set(cwise_anno[class_chosen])
            num_file = len(file_class_chosen)

            

            ## same class for S+Q : todo : support label and query label is same
            for k in range(nshot):
                support_index = random.randint(1, num_file) - 1 ## -1 because one sample is already taken in query fom indexes
                support_vname = cwise_anno[class_chosen][support_index]
                support_vid_list.append(support_vname)
                support_data = self.loadFeaturecwise(support_vname)
                support_label = self.getVideoMaskcwise(self.annos_label, support_vname, sub_cls_dict)
                support_label_class,support_label_loc, support_label_mask, sp_lb_cgt, sp_lb_mgt, strt_idx_s, end_idx_s = self.getBranchData(support_label)               
                support_data_zero = torch.zeros_like(support_data)
                
                if self.is_trim:
                    for i in range(len(strt_idx)):
                        support_data_zero[:,strt_idx_s[i]+1:end_idx_s[i]-1] = support_data[:,strt_idx_s[i]+1:end_idx_s[i]-1]
                    supp_mask = support_data_zero !=0
                    support_data_trimmed = (support_data_zero*supp_mask).sum(dim=1)/supp_mask.sum(dim=1)
                    support_data_trimmed = support_data_trimmed.unsqueeze(1).expand(-1,support_data.size(1))
                else:
                    support_data_trimmed = support_data_zero
                
                
                support_video_list.append(support_data)
                support_video_list_trim.append(support_data_trimmed)
                support_label_list_loc.append(support_label_loc)
                support_label_list_class.append(support_label_class)
                support_label_list_mask.append(support_label_mask)
                support_label_list_cgt.append(sp_lb_cgt)
                support_label_list_mgt.append(sp_lb_mgt)

            assert len(support_video_list) == nshot

            #### store the support data and labels overall shot ####

            spprt_vids = torch.cat(support_video_list,0).view(nshot,T,C)
            spprt_vids_trim = torch.cat(support_video_list_trim,0).view(nshot,T,C)
            cls_support_video.append(spprt_vids)
            cls_support_video_trim.append(spprt_vids_trim)
            cls_support_label_loc.append(torch.cat(support_label_list_loc,0).view(nshot,T,T))
            cls_support_label_class.append(torch.cat(support_label_list_class,0).view(nshot,-1))
            cls_support_label_mask.append(torch.cat(support_label_list_mask,0).view(nshot,nway+1,T))
            cls_support_label_cgt.append(torch.cat(support_label_list_cgt,0).view(nshot,-1))
            cls_support_label_mgt.append(torch.cat(support_label_list_mgt,0).view(nshot,-1))

            #### store the query data and labels overall shot ####

            cls_query_video.append(query_data)
            cls_query_video_trim.append(query_data_trimmed)
            cls_query_label_mask.append(query_label_mask)
            cls_query_label_loc.append(query_label_loc)
            cls_query_label_class.append(query_label_class)
            cls_query_label_cgt.append(q_lb_cgt)
            cls_query_label_mgt.append(q_lb_mgt)

        
        #### store the support data and labels overall class ####
        support_data_tensor = torch.cat(cls_support_video,0).view(nway,nshot,C,T)  ### [nway=batch,nshot,feat_dim,temp]
        support_data_trim_tensor = torch.cat(cls_support_video_trim,0).view(nway,nshot,C,T)
        support_class_tensor = torch.cat(cls_support_label_class,0).view(nway,nshot,-1)
        support_loc_tensor = torch.cat(cls_support_label_loc,0).view(nway,nshot,T,T)
        support_mask_tensor = torch.cat(cls_support_label_mask,0).view(nway,nshot,nway+1,T)
        support_cgt_tensor = torch.cat(cls_support_label_cgt,0).view(nway,nshot,-1)
        support_mgt_tensor = torch.cat(cls_support_label_mgt,0).view(nway,nshot,-1)

        if self.is_trim:
            support_dict = {
                'data':support_data_trim_tensor,
                # 'trim_data':support_data_trim_tensor,
                'class_branch':support_class_tensor,
                'mask_branch':support_loc_tensor,
                '1d_mask':support_mgt_tensor,
                'class_branch_bin':support_mask_tensor,
                'class_1d_gt':support_cgt_tensor
            }
        else:
            support_dict = {
            'data':support_data_tensor,
            # 'trim_data':support_data_trim_tensor,
            'class_branch':support_class_tensor,
            'mask_branch':support_loc_tensor,
            '1d_mask':support_mgt_tensor,
            'class_branch_bin':support_mask_tensor,
            'class_1d_gt':support_cgt_tensor
        }
        
        #### store the query data and labels overall class ####
        
        query_data_tensor = torch.cat(cls_query_video,0).view(nway,-1) ### [batch,nway,feat_dim,temp]
        query_data_trim_tensor = torch.cat(cls_query_video_trim,0).view(nway,-1)
        query_class_tensor = torch.cat(cls_query_label_class,0).view(nway,-1)
        query_loc_tensor = torch.cat(cls_query_label_loc,0).view(nway,T,T)
        query_mask_tensor = torch.cat(cls_query_label_mask,0).view(nway,nway+1,T)
        query_cgt_tensor = torch.cat(cls_query_label_cgt,0).view(nway,-1)
        query_mgt_tensor = torch.cat(cls_query_label_mgt,0).view(nway,-1)    

        if self.is_trim:
            query_dict = {
                'data':query_data_trim_tensor,
                # 'data_trim':query_data_trim_tensor,
                'class_branch':query_class_tensor,
                'mask_branch':query_loc_tensor,
                '1d_mask':query_mgt_tensor,
                'class_branch_bin':query_mask_tensor,
                'class_1d_gt':query_cgt_tensor
            } 
        else:
            query_dict = {
                'data':query_data_tensor,
                # 'data_trim':query_data_trim_tensor,
                'class_branch':query_class_tensor,
                'mask_branch':query_loc_tensor,
                '1d_mask':query_mgt_tensor,
                'class_branch_bin':query_mask_tensor,
                'class_1d_gt':query_cgt_tensor
            }  

        meta_data_dict ={
            'support_video_id':support_vid_list,
            'query_video_id':query_vid_list,
            'class_list': rand_cls_list
        }

        return support_dict, query_dict, meta_data_dict

        
    def get_video_anno(self,video_infos,video_anno_path):

        anno_database = load_json(self.video_anno_path)
        # print(anno_database)
        video_dict = {}
        for video_name in video_infos.keys():
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
                    clip_factor = self.temporal_scale / ( corr_sec * (self.num_frame+1) )
                    action_start = tmp_info['segment'][0]*clip_factor
                    snip_start = max(min(1, tmp_info['segment'][0] / corr_sec), 0)
                    action_end = tmp_info['segment'][1]*clip_factor
                    snip_end = max(min(1, tmp_info['segment'][1] / corr_sec), 0)
                    gt_label = tmp_info["label"]

                if action_end - action_start > 1 and gt_label in test_class:
                    label_list.append([snip_start,snip_end,gt_label])

            if len(label_list)>0:
                temporal_dict[idx]= {"labels":label_list,
                                    "video_duration": num_sec}
            
        return temporal_dict

    def getVideoMask(self,video_annos,clip_length=100):

        self.video_mask = {}
        idx_list = self.getAnnotation(self.subset,video_annos)
        print("No of videos in "+ self.subset + " is "+ str(len(idx_list.keys())))
        self.anno_final = idx_list
        self.anno_final_idx = list(idx_list.keys())
        print('Loading '+self.subset+' Video Information ...')
        print("No of class", len(self.class_to_idx.keys()))
        for idx in tqdm.tqdm(list(video_annos.keys()),ncols=0):
            if os.path.exists(os.path.join(self.feature_path+"/",idx+".npy")) and idx in list(idx_list.keys()):
                cur_annos = idx_list[idx]["labels"]
                mask_list=[]
                for l_id in range(len(cur_annos)):
                    mask_start = int(math.floor(clip_length*cur_annos[l_id][0]))
                    mask_end = int(math.floor(clip_length*cur_annos[l_id][1]))
                    mask_label_idx = self.class_to_idx[cur_annos[l_id][2]]
                    mask_list.append([mask_start,mask_end,mask_label_idx])
                self.video_mask[idx] = mask_list

        return self.video_mask
                
    def loadFeature(self, idx):

        feat_path = self.feature_path
        feat = np.load(os.path.join(feat_path, idx+".npy"))
        feat_tensor = torch.Tensor(feat)
        video_data = torch.transpose(feat_tensor, 0, 1)
        video_data = F.interpolate(video_data.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
        return video_data


    def getVideoData(self,index):

        mask_idx = self.subset_mask_list[index]
        mask_data = self.loadFeature(mask_idx)
        mask_label = self.video_mask[mask_idx]
        mask_data_trimmed = torch.zeros_like(mask_data)

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
          mask_data_trimmed[:,mod_start:mod_end] = mask_data[:,mod_start:mod_end]
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
        fg_mask = torch.Tensor(temporary_mask)
        bg_mask = torch.Tensor(background_mask)
        bot_mask = fg_mask
        v_label = torch.Tensor()

        return mask_data, classifier_branch,global_mask_branch,mask_top,cas_mask,bot_mask


    def __getitem__(self, index):

        support_dict, query_dict, meta_dict = self.getEpisodicVideoData(index)
        
        return index, support_dict, query_dict, meta_dict






    def __len__(self):
        return len(self.subset_mask_list)

