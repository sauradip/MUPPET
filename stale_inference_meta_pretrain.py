

import os
import math
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import itertools,operator
from stale_model_fs import STALE as STALEFS  ## STALE novel class
import stale_lib.stale_dataloader_base_pretrain as stale_dataset
import stale_lib.stale_dataloader_fs_pretrain as stale_dataset_fs
from scipy import ndimage
from scipy.special import softmax
from collections import Counter
import cv2
import json
from config.dataset_class import activity_dict
import yaml
from utils.postprocess_utils import multithread_detection , get_infer_dict, load_json
from joblib import Parallel, delayed
from config.dataset_class import activity_dict
from config.few_shot import base_class,val_class,test_class,base_dict,val_dict,test_dict
from tqdm import tqdm
import random
from stale_lib.loss_stale_fs import stale_loss


with open("./config/anet_2gpu.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

if __name__ == '__main__':


    output_path = config['dataset']['testing']['output_path']
    ## few-shot setting ##
    fsmode = config['fewshot']['mode']
    nepisode = config['fewshot']['episode']
    nshot = config['fewshot']['shot']
    emb_dim = config['pretraining']['emb_dim']
    is_postprocess = True
    if not os.path.exists(output_path + "/results"):
        os.makedirs(output_path + "/results")

    ### Load Model ###
    model = STALEFS()
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4]).cuda()
    ### Load Base Class trained Model Checkpoint ###
    print(' -- debug', output_path + "/STALE_base_best.pth.tar")
    checkpoint = torch.load(output_path + "/STALE_base_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'],strict=False)

    def post_process_multi(detection_thread,get_infer_dict):

        nms_thres = config['testing']['nms_thresh']
        
        infer_dict , label_dict = get_infer_dict()
        pred_data = pd.read_csv("stale_output.csv")
        pred_videos = list(pred_data.video_name.values[:])
        cls_data_score, cls_data_cls = {}, {}
        best_cls = load_json("stale_best_score.json")
        
        for idx, vid in enumerate(infer_dict.keys()):
            if vid in pred_videos:
                vid = vid[2:] 
                cls_data_cls[vid] = best_cls["v_"+vid]["class"] 

        parallel = Parallel(n_jobs=15, prefer="processes")
        detection = parallel(delayed(detection_thread)(vid, video_cls, infer_dict['v_'+vid], label_dict, pred_data,best_cls)
                            for vid, video_cls in cls_data_cls.items())
        detection_dict = {}
        [detection_dict.update(d) for d in detection]
        output_dict = {"version": "ANET v1.3, STALE", "results": detection_dict, "external_data": {}}

        with open(output_path + '/detection_result_nms{}.json'.format(nms_thres), "w") as out:
            json.dump(output_dict, out)
    
    def generate_prediction(top_br_pred,bottom_br_pred,new_props,supp_cls_dict,video_name):

        key_list = list(supp_cls_dict.keys())
        val_list = list(supp_cls_dict.values())
        tscale = 100
        num_class = len(key_list)
        # print(len(key_list))

        # print(new_props)

        top_k_snip = config['testing']['top_k_snip']
        
        class_snip_thresh = config['testing']['class_thresh']
        mask_snip_thresh = config['testing']['mask_thresh']
        tscale = config['model']['temporal_scale']


        props = bottom_br_pred[0].detach().cpu().numpy()

        ### classifier branch prediction ###

        soft_cas = torch.softmax(top_br_pred[0],dim=0) 
        # print(soft_cas)
        # soft_cas_topk,soft_cas_topk_loc = torch.topk(soft_cas[:num_class],2,dim=0)
        top_br_np = softmax(top_br_pred[0].detach().cpu().numpy(),axis=0)[:num_class]

        label_pred = torch.softmax(torch.mean(top_br_pred[0][:num_class,:],dim=1),axis=0).detach().cpu().numpy()
        vid_label_id = np.argmax(label_pred)
        vid_label_sc = np.amax(label_pred)
        # print(vid_label_id)
        props_mod = props[props>0]
        top_br_np = softmax(top_br_pred[0].detach().cpu().numpy(),axis=0)[:num_class]

        top_br_mean = np.mean(top_br_np,axis=1)
        top_br_mean_max = np.amax(top_br_np,axis=1)
        top_br_mean_id = np.argmax(top_br_mean)
        
        

        soft_cas_np = soft_cas[:num_class].detach().cpu().numpy()
        seg_score = np.zeros([tscale])
        seg_cls = []
        seg_mask = np.zeros([tscale])

        ### for each snippet, store the max score and class info ####

        for j in range(tscale):
            
            seg_score[j] =  np.amax(soft_cas_np[:,j])
            seg_cls.append(np.argmax(soft_cas_np[:,j]))

        # seg_score[seg_score < class_thres] = 0

        thres = class_snip_thresh

        cas_tuple = []
        for k in thres:
            filt_seg_score = seg_score > k
            integer_map1 = map(int,filt_seg_score)
            filt_seg_score_int = list(integer_map1)
            filt_seg_score_int = ndimage.binary_fill_holes(filt_seg_score_int).astype(int).tolist()
            if 1 in filt_seg_score_int:
                start_pt1 = filt_seg_score_int.index(1)
                end_pt1 = len(filt_seg_score_int) - 1 - filt_seg_score_int[::-1].index(1)

                if end_pt1 - start_pt1 > 1:
                    scores = np.amax(seg_score[start_pt1:end_pt1])
                    label = max(set(seg_cls[start_pt1:end_pt1]), key=seg_cls.count)
                    cas_tuple.append([start_pt1,end_pt1,scores,label])

        max_score, score_idx  = torch.max(soft_cas[:num_class],0)
        soft_cas_np = soft_cas[:num_class].detach().cpu().numpy()
        score_map = {}

        top_np = top_br_pred[0][:num_class].detach().cpu().numpy()  
        top_np_max = np.mean(top_np,axis=1)
        max_score_np = max_score.detach().cpu().numpy()
        score_idx = score_idx.detach().cpu().numpy()

        for ids in range(len(score_idx)):
            score_map[max_score_np[ids]]= score_idx[ids]

        
        k = top_k_snip ## more fast inference
        max_idx = np.argpartition(max_score_np, -k)[-k:]


        ### indexes of top K scores ###

        top_k_idx = max_idx[np.argsort(max_score_np[max_idx])][::-1].tolist()

        for locs in top_k_idx:

            seq = props[locs,:]
            thres = mask_snip_thresh

            for j in thres:
                filtered_seq = seq > j
            
                integer_map = map(int,filtered_seq)
                filtered_seq_int = list(integer_map)
                filtered_seq_int2 = ndimage.binary_fill_holes(filtered_seq_int).astype(int).tolist()
                
                if 1 in filtered_seq_int:

                    #### getting start and end point of mask from mask branch ####

                    start_pt1 = filtered_seq_int2.index(1)
                    end_pt1 = len(filtered_seq_int2) - 1 - filtered_seq_int2[::-1].index(1) 
                    r = max((list(y) for (x,y) in itertools.groupby((enumerate(filtered_seq_int)),operator.itemgetter(1)) if x == 1), key=len)
                    start_pt = r[0][0]
                    end_pt = r[-1][0]
                    if (end_pt - start_pt)/tscale > 0.02 : 
                    #### get (start,end,cls_score,reg_score,label) for each top-k snip ####

                        score_ = max_score_np[locs]
                        cls_score = score_
                        lbl_id = score_map[score_]
                        reg_score = np.amax(seq[start_pt+1:end_pt-1])
                        label = key_list[val_list.index(lbl_id)]
                        vid_label = key_list[val_list.index(vid_label_id)]
                        score_shift = np.amax(soft_cas_np[vid_label_id,start_pt:end_pt])
                        prop_start = start_pt1/tscale
                        prop_end = end_pt1/tscale
                        new_props.append([video_name, prop_start , prop_end , score_shift*reg_score, score_shift*cls_score,vid_label])     
                        
                        for m in range(len(cas_tuple)):
                            start_m = cas_tuple[m][0]
                            end_m = cas_tuple[m][1]
                            score_m = cas_tuple[m][2]
                            reg_score = np.amax(seq[start_m:end_m])
                            prop_start = start_m/tscale
                            prop_end = end_m/tscale
                            cls_score = score_m

                            new_props.append([video_name, prop_start,prop_end,reg_score,cls_score,vid_label])

        return new_props



    ### Load Dataloader ###
    if fsmode == 1: ### base evaluation
        test_loader = torch.utils.data.DataLoader(stale_dataset.STALEDataset(subset="validation", mode='inference'),
                                                batch_size=1, shuffle=False,
                                                num_workers=8, pin_memory=True, drop_last=False)
        model.eval()
    elif fsmode == 2 or fsmode == 3: ### meta evaluation
        test_loader = torch.utils.data.DataLoader(stale_dataset_fs.STALEEpisodicDataset(subset="validation", mode='inference'),
                                                batch_size=1, shuffle=False,
                                                num_workers=8, pin_memory=False, drop_last=False)


    if fsmode == 1:
        #### to do base class inference #####
        print("later --> Not priority")
    elif fsmode == 2:

        #### to do meta-train #### 

        batch_size_val = 1 # consumes gpu-ram
        is_trim = config['fewshot']['trimmed']
        supp_trim = config['fewshot']['trim_support']
        # shot = 
        norm_feat = True
        n_runs = 1
        episodes = 250 
        # lr = 0.0004
        for param in model.parameters():
                param.requires_grad = True
                    
        model.module.embedding.requires_grad = False
        model.module.txt_model.requires_grad = True
        # model.module.masktrans.requires_grad = False
        model.module.localizer_mask.requires_grad = False
        model.train()

        model_param = list(model.parameters()) ## getting the params from the transformer linear layers for optimization
        optimizer_model = torch.optim.Adam(model_param, lr=0.00004)
        best_loss = 0
        # tscale = 100
        tscale = config['model']['temporal_scale']
        # ========== Perform the runs  ==========
        for run in range(n_runs): ## epochs
            for e in tqdm(range(episodes)):
                #### N-way K-shot : support set used for training the new (N+1) classifier and query set is used for evalutation the classifier to save model
                iter_loader = iter(test_loader)
                ### to do : complete structure, change model to accept the class list so that N+1 can be created
                for i in range(batch_size_val):
                    try:
                        idx, support_dict, query_dict, meta_dict = iter_loader.next()
                    except:
                        iter_loader = iter(test_loader)
                        idx, support_dict, query_dict, meta_dict = iter_loader.next()

                    support_data = support_dict['data'][0]
                    nway,nshot,C,T,H,W = support_data.size()
                    support_top_gt = support_dict['class_branch'][0].view(-1,tscale)
                    support_bot_gt = support_dict['mask_branch'][0].view(-1,tscale,tscale)
                    support_1d_gt = support_dict['1d_mask'][0].view(-1,tscale)
                    support_class_bin = support_dict['class_branch_bin'][0].view(-1,nway+1,tscale)
                    support_class_gt = support_dict['class_1d_gt'].view(-1,nway)

                    query_data = query_dict['data'][0].view(-1,C,T,H,W)
                    query_top_gt = query_dict['class_branch'][0].view(-1,tscale)
                    query_bot_gt = query_dict['mask_branch'][0].view(-1,tscale,tscale)
                    query_1d_gt = query_dict['1d_mask'][0].view(-1,tscale)
                    query_class_bin = query_dict['class_branch_bin'][0].view(-1,nway+1,tscale)
                    query_class_gt = query_dict['class_1d_gt'][0].view(-1,nway)

                    subcls = meta_dict['class_list']
                    supp_cls_dict = {}
                    supp_cls = []
                    for i in range(nway):
                        supp_cls.append(subcls[i][0])
                        supp_cls_dict[subcls[i][0]] = i
                    support_data = support_data.view(-1,C,T,H,W)
                    support_data_nshot = support_data.view(-1,nshot,C,T,H,W) # [nway,nshot,C,T]
                    support_1d_gt_nshot = support_1d_gt.view(-1,nshot,tscale) # [nway,nshot,T]#
                    support_1d_gt_nshot = support_1d_gt_nshot.unsqueeze(2).expand(-1,-1,emb_dim,-1) # [nway,nshot,C,T]
                    
                    ##### meta-learning data #####
                    if supp_trim:
                        support_trimmed = support_data_nshot*support_1d_gt_nshot # [nway, nshot, C, T] 
                    else:
                        support_trimmed = support_data_nshot
                    # support_proto = torch.mean(support_trimmed,dim=1) # [nway, C, T]
                    # support_proto = torch.mean(support_proto, dim=2) # [nway,C]

                    support_proto = support_1d_gt_nshot


                    # support_top_pred, support_bot_pred, support_1d_pred = model(support_data,query_data,supp_cls,support_proto,support_trimmed,mode="train")
                    # supp_loss_train = gsm_loss(support_top_gt,support_top_pred,support_bot_gt,support_bot_pred,support_class_bin,support_1d_pred,support_1d_gt,"train")

                    support_top_pred, support_bot_pred, support_1d_pred, support_cls_pred, features = model(support_data,query_data,supp_cls,support_proto,support_trimmed,mode="train")
                    supp_loss_train = stale_loss(support_top_gt,support_top_pred,support_bot_gt,support_bot_pred,support_class_bin,support_1d_pred,support_1d_gt,support_cls_pred,support_class_gt,features,"train")
                    tot_loss_supp = supp_loss_train[0]

                    print("[Episode {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f} + M-Loss {4:.2f} (train)".format(
    e, tot_loss_supp,supp_loss_train[1],supp_loss_train[2],supp_loss_train[3]))

                    optimizer_model.zero_grad()
                    tot_loss_supp.backward()
                    optimizer_model.step()

                    # query_top_pred, query_bot_pred, query_1d_pred = model(query_data,support_data,supp_cls,support_proto,support_trimmed,mode="test")
                    # query_loss_val = gsm_loss(query_top_gt,query_top_pred,query_bot_gt,query_bot_pred,query_class_bin,query_1d_pred,query_1d_gt,mode="test")


                    query_top_pred, query_bot_pred, query_1d_pred, query_cls_pred, features = model(query_data,support_data,supp_cls,support_proto,support_trimmed,mode="test")
                    query_loss_val = stale_loss(query_top_gt,query_top_pred,query_bot_gt,query_bot_pred,query_class_bin,query_1d_pred,query_1d_gt,query_cls_pred,query_class_gt,features,mode="test")
                    tot_loss_query = query_loss_val[0]

                    print("[Episode {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f} + M-Loss {4:.2f} (test)".format(
    e, tot_loss_query,query_loss_val[1],query_loss_val[2],query_loss_val[3]))

                    state = {'epoch': e + 1,
                            'state_dict': model.state_dict()}
                    torch.save(state, output_path + "/STALE_meta_support_checkpoint.pth.tar")

            ####### Once Trained with Support Examples , Train with Query Samples #######

            checkpoint = torch.load(output_path + "/STALE_meta_support_checkpoint.pth.tar")
            model.load_state_dict(checkpoint['state_dict'])

            for param in model.parameters():
                        param.requires_grad = True
                    
            model.module.embedding.requires_grad = True
            model.module.txt_model.requires_grad = False
            model.module.masktrans.requires_grad = False
            model.module.localizer_mask.requires_grad = False
            model.train()
            tscale = config['model']['temporal_scale']
            for e in tqdm(range(episodes)):
                #### N-way K-shot : support set used for training the new (N+1) classifier and query set is used for evalutation the classifier to save model
                iter_loader = iter(test_loader)
                ### to do : complete structure, change model to accept the class list so that N+1 can be created
                for i in range(batch_size_val):
                    try:
                        idx, support_dict, query_dict, meta_dict = iter_loader.next()

                    except:
                        iter_loader = iter(test_loader)
                        idx, support_dict, query_dict, meta_dict = iter_loader.next()

                    support_data = support_dict['data'][0]
                    nway,nshot,C,T,H,W = support_data.size()
                    support_top_gt = support_dict['class_branch'][0].view(-1,tscale)
                    # print(support_top_gt.size())
                    support_bot_gt = support_dict['mask_branch'][0].view(-1,tscale,tscale)
                    support_1d_gt = support_dict['1d_mask'][0].view(-1,tscale)
                    support_class_bin = support_dict['class_branch_bin'][0].view(-1,nway+1,tscale)
                    support_class_gt = support_dict['class_1d_gt'].view(-1,nway)

                    query_data = query_dict['data'][0].view(-1,C,T,H,W)
                    query_top_gt = query_dict['class_branch'][0].view(-1,tscale)
                    query_bot_gt = query_dict['mask_branch'][0].view(-1,tscale,tscale)
                    query_1d_gt = query_dict['1d_mask'][0].view(-1,tscale)
                    query_class_bin = query_dict['class_branch_bin'][0].view(-1,nway+1,tscale)
                    query_class_gt = query_dict['class_1d_gt'][0].view(-1,nway)

                    subcls = meta_dict['class_list']
                    supp_cls_dict = {}
                    supp_cls = []
                    for i in range(nway):
                        supp_cls.append(subcls[i][0])
                        supp_cls_dict[subcls[i][0]] = i

                    support_data = support_data.view(-1,C,T,H,W)
                    support_data_nshot = support_data.view(-1,nshot,C,T,H,W) # [nway,nshot,C,T]
                    support_1d_gt_nshot = support_1d_gt.view(-1,nshot,tscale) # [nway,nshot,T]#
                    support_1d_gt_nshot = support_1d_gt_nshot.unsqueeze(2).expand(-1,-1,emb_dim,-1) # [nway,nshot,C,T]

                    ##### meta-learning data #####
                    if supp_trim:
                        support_trimmed = support_data_nshot*support_1d_gt_nshot # [nway, nshot, C, T]
                        
                    else:
                        support_trimmed = support_data_nshot
                    # print(support_proto.size())
                    # support_proto = torch.mean(support_trimmed,dim=1) # [nway, C, T]
                    # support_proto = torch.mean(support_proto, dim=2) # [nway,C]

                    support_proto = support_1d_gt_nshot
                    query_top_pred, query_bot_pred, query_1d_pred, query_cls_pred, features = model(query_data,support_data,supp_cls,support_proto,support_trimmed,mode="test")

                    query_loss_val = stale_loss(query_top_gt,query_top_pred,query_bot_gt,query_bot_pred,query_class_bin,query_1d_pred,query_1d_gt,query_cls_pred,query_class_gt,features,mode="test")
                    tot_loss_query = query_loss_val[0]
                    print("[Episode {0:03d}] Total-Loss {1:.2f} = T-Loss {2:.2f} + B-Loss {3:.2f} + M-Loss {4:.2f} (test)".format(
    e, tot_loss_query,query_loss_val[1],query_loss_val[2],query_loss_val[3]))

                    state = {'epoch': e + 1,
                            'state_dict': model.state_dict()}
                    torch.save(state, output_path + "/STALE_meta_adaptation_checkpoint.pth.tar")

                    if tot_loss_query < best_loss : 
                        best_loss = tot_loss_query
                        torch.save(state, output_path + "/STALE_meta_adaptation_best_checkpoint.pth.tar")  
    else:

        ### to do meta-test ###
        print("later")
        batch_size_val = 1 # consumes gpu-ram
        # shot = 
        norm_feat = True
        n_runs = 1
        episodes = 50
        supp_trim = config['fewshot']['trim_support']

        tscale = config['model']['temporal_scale']
        new_props = list()
        checkpoint = torch.load(output_path + "/STALE_meta_adaptation_checkpoint.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        file = "stale_output.csv"
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)
        print("Inference start")
        for run in range(n_runs): ## epochs
            for e in tqdm(range(episodes)):
                #### N-way K-shot : support set used for training the new (N+1) classifier and query set is used for evalutation the classifier to save model
                iter_loader = iter(test_loader)
                ### to do : complete structure, change model to accept the class list so that N+1 can be created
                for i in range(batch_size_val):
                    try:
                        idx, support_dict, query_dict, meta_dict = iter_loader.next()
                    except:
                        iter_loader = iter(test_loader)
                        idx, support_dict, query_dict, meta_dict = iter_loader.next()

                    support_data = support_dict['data'][0]
                    nway,nshot,C,T,H,W = support_data.size()
                    support_top_gt = support_dict['class_branch'][0].view(-1,tscale)
                    support_bot_gt = support_dict['mask_branch'][0].view(-1,tscale,tscale)
                    support_1d_gt = support_dict['1d_mask'][0].view(-1,tscale)
                    support_class_bin = support_dict['class_branch_bin'][0].view(-1,nway+1,tscale)
                    support_class_gt = support_dict['class_1d_gt'].view(-1,nway)

                    query_data = query_dict['data'][0].view(-1,C,T,H,W)
                    query_top_gt = query_dict['class_branch'][0].view(-1,tscale)
                    query_bot_gt = query_dict['mask_branch'][0].view(-1,tscale,tscale)
                    query_1d_gt = query_dict['1d_mask'][0].view(-1,tscale)
                    query_class_bin = query_dict['class_branch_bin'][0].view(-1,nway+1,tscale)
                    query_class_gt = query_dict['class_1d_gt'][0].view(-1,nway)

                    subcls = meta_dict['class_list']
                    query_vid = meta_dict['query_video_id']
                    # print(query_vid)
                    supp_cls = []
                    query_vid_new = []
                    supp_cls_dict = {}
                    for i in range(nway):
                        supp_cls.append(subcls[i][0])
                        supp_cls_dict[subcls[i][0]] = i
                        query_vid_new.append(query_vid[i][0])

                    ##### meta-learning data #####
                    support_data = support_data.view(-1,C,T,H,W)
                    support_data_nshot = support_data.view(-1,nshot,C,T,H,W) # [nway,nshot,C,T]
                    support_1d_gt_nshot = support_1d_gt.view(-1,nshot,tscale) # [nway,nshot,T]#
                    support_1d_gt_nshot = support_1d_gt_nshot.unsqueeze(2).expand(-1,-1,emb_dim,-1) # [nway,nshot,C,T]

                    if supp_trim:
                        support_trimmed = support_data_nshot*support_1d_gt_nshot # [nway, nshot, C, T]
                    else:
                        support_trimmed = support_data_nshot

#                     support_proto = torch.mean(support_trimmed,dim=1) # [nway, C, T]
#                     support_proto = torch.mean(support_proto, dim=2) # [nway,C]
                    support_proto = support_1d_gt_nshot
                    query_top_pred, query_bot_pred, query_1d_pred, query_cls_pred, features = model(query_data,support_data,supp_cls,support_proto,support_trimmed,mode="test")
                    # print()
                    mult,cl,T = query_top_pred.size()
                    # print(query_vid_new)
                    # print(nway)
                    for i in range(nway):
                        # for j in range(nshot):
                            vid_id = query_vid_new[i]
                            top_pred = query_top_pred[i,:,:].view(1,cl,T)
                            bot_pred = query_bot_pred[i,:,:].view(1,T,T)
                            new_props = generate_prediction(top_pred,bot_pred,new_props,supp_cls_dict,vid_id)

            ### filter duplicate proposals --> Less Time for Post-Processing #####
            new_props1 = np.stack(new_props)
            b_set = set(map(tuple,new_props1))  
            result = map(list,b_set) 

            ### save the proposals in a csv file ###
            col_name = ["video_name","xmin", "xmax", "clr_score", "reg_score","label"]
            new_df = pd.DataFrame(result, columns=col_name)
            new_df.to_csv("stale_output.csv", index=False)

        ###### Post-Process #####
        print("Start Post-Processing")
        post_process_multi(multithread_detection,get_infer_dict)
        print("End Post-Processing")

    
        
