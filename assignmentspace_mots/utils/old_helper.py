
import numpy as np
from collections import namedtuple

import munkres

import pycocotools.mask as cocomask
import argparse
import os
import math
import matplotlib.pyplot as plt
#from wasserstein_check import *
import time
import sys
import dp_mots as dp  # dyn_prog,get_assignment_cost_matrix,assignment_dyn_prog
import torch
#import helper_mots as helper
sys.path.append('../')
from HungarianMurty.HungarianMurty_lowmem import k_best_costs
import networkx as nx
import pdb
#sys.path.append('../../../../OpticalFlow/PWC-Net/PyTorch')

TrackElement_ = namedtuple("TrackElement", ["t", "box", "reid", "track_id", "class_", "mask", "score"])
TrackElement = namedtuple("TrackElement", ["box", "track_id", "class_", "mask", "score"])

munkres_obj = munkres.Munkres()

tracker_options = {"tracker": "hungarian", "reid_comp": "euclidean",
                   "detection_confidence_threshold_car": 0.843,
                   # min of what is given in segtrack_tune_experiment.py, option: mask
                   "detection_confidence_threshold_pedestrian": 0.936,
                   # min of what is given in segtrack_tune_experiment.py, option: mask
                   "reid_weight_car": 0.0,
                   "reid_weight_pedestrian": 0.0,
                   "mask_iou_weight_car": 1.0,
                   "mask_iou_weight_pedestrian": 1.0,
                   "bbox_center_weight_car": 0.0,
                   "bbox_center_weight_pedestrian": 0.0,
                   "bbox_iou_weight_car": 0.0,
                   "bbox_iou_weight_pedestrian": 0.0,
                   "association_threshold_car": 0.0163,  # check segtrack_tune_experiment.py
                   "association_threshold_pedestrian": 0.00098,  # check segtrack_tune_experiment.py
                   "keep_alive_car": 0,
                   "keep_alive_pedestrian": 0,
                   "reid_euclidean_offset_car": 5.0,
                   "reid_euclidean_scale_car": 1.0,
                   "reid_euclidean_offset_pedestrian": 5.0,
                   "reid_euclidean_scale_pedestrian": 1.0,
                   "new_reid_threshold_car": 2.0,
                   "new_reid_threshold_pedestrian": 2.0,
                   "box_offset": 50.0,
                   "box_scale": 0.02,
                   "new_reid": False}


def do_tracking(tracking_weights, images, labels, predictions, seq, path, image_names,
                     start_time_at_1=False, training=True):
    # labels can be [] in case of inference
    boxes=predictions["boxes"]
    scores=predictions["scores"]
    reids=predictions["reids"]
    classes=predictions["classes"]
    masks=predictions["masks"]
    track_loss=torch.tensor([0.]).cuda()

    gt_track_ids=[[p["track_id"] for p in q] for q in labels]




    structured_mots_options = {"K": 20}
    structured_mots_options["leaf"]=0
    structured_mots_options["second_order"] = 0
    structured_mots_options["second_order_weight"] = 0
    structured_mots_options["lambda_iou_main"]=tracking_weights[0]
    structured_mots_options["lambda_app_main"]=tracking_weights[1]

    #optical_flow_skip0, optical_flow_skip1 = get_optical_flow(path, seq, image_names)

    hyp_tracks, track_loss = track_single_sequence(tracker_options, boxes, scores, reids, classes, masks, structured_mots_options, labels, optical_flow0=None, optical_flow1=None,
                                       images=images, unique_classes=[0,1]) #0 for car, 1 for pedestrian

    return hyp_tracks, track_loss

def track_single_sequence(tracker_options, boxes, scores, reids, classes, masks, structured_mots_options,labels,
                          optical_flow0=None, optical_flow1=None, images=None, unique_classes=[0,1], training=True):
    # perform tracking per class and in the end combine the results

    start_track_id = 1
    class_tracks = []
    tracker_options_class = {"tracker": tracker_options["tracker"], "reid_comp": tracker_options["reid_comp"],
                             "box_offset": tracker_options["box_offset"],
                             "box_scale": tracker_options["box_scale"]}
    for class_ in unique_classes:
        if class_ == unique_classes[0]:
            tracker_options_class["detection_confidence_threshold"] = tracker_options[
                "detection_confidence_threshold_car"]
            tracker_options_class["reid_weight"] = tracker_options["reid_weight_car"]
            tracker_options_class["mask_iou_weight"] = tracker_options["mask_iou_weight_car"]
            tracker_options_class["bbox_iou_weight"] = tracker_options["bbox_iou_weight_car"]
            tracker_options_class["bbox_center_weight"] = tracker_options["bbox_center_weight_car"]
            tracker_options_class["association_threshold"] = tracker_options["association_threshold_car"]
            tracker_options_class["keep_alive"] = tracker_options["keep_alive_car"]
            tracker_options_class["new_reid_threshold"] = tracker_options["new_reid_threshold_car"]
            tracker_options_class["reid_euclidean_offset"] = tracker_options["reid_euclidean_offset_car"]
            tracker_options_class["reid_euclidean_scale"] = tracker_options["reid_euclidean_scale_car"]
        elif class_ == unique_classes[1]:
            tracker_options_class["detection_confidence_threshold"] = tracker_options[
                "detection_confidence_threshold_pedestrian"]
            tracker_options_class["reid_weight"] = tracker_options["reid_weight_pedestrian"]
            tracker_options_class["mask_iou_weight"] = tracker_options["mask_iou_weight_pedestrian"]
            tracker_options_class["bbox_iou_weight"] = tracker_options["bbox_iou_weight_pedestrian"]
            tracker_options_class["bbox_center_weight"] = tracker_options["bbox_center_weight_pedestrian"]
            tracker_options_class["association_threshold"] = tracker_options["association_threshold_pedestrian"]
            tracker_options_class["keep_alive"] = tracker_options["keep_alive_pedestrian"]
            tracker_options_class["new_reid_threshold"] = tracker_options["new_reid_threshold_pedestrian"]
            tracker_options_class["reid_euclidean_offset"] = tracker_options["reid_euclidean_offset_pedestrian"]
            tracker_options_class["reid_euclidean_scale"] = tracker_options["reid_euclidean_scale_pedestrian"]
        else:
            assert False, "unknown class"
        if training:
            assignment_list, track_loss= tracker_per_class(tracker_options_class, boxes, scores, reids, classes, masks, class_,
                             structured_mots_options, labels,
                             optical_flow0=optical_flow0, optical_flow1=optical_flow1, images=images)
            return assignment_list, track_loss
        else:
            tracks=[]
            if [class_] in classes:
                tracks = tracker_per_class(tracker_options_class, boxes, scores, reids, classes, masks, class_,
                                       structured_mots_options,
                                       optical_flow0=optical_flow0, optical_flow1=optical_flow1, images=images)
    return tracks


def tracker_per_class(tracker_options, boxes, scores, reids, classes, masks, class_to_track, structured_mots_options, labels,
                      optical_flow0=None, optical_flow1=None, images=None, training=True):

    if optical_flow0 is None:
        optical_flow0 = [None for _ in masks]
    else:
        optical_flow0 = [None] + optical_flow0
        assert len(boxes) == len(scores) == len(reids) == len(classes) == len(masks) == len(optical_flow0)
    if optical_flow1 is None:
        optical_flow1 = [None for _ in masks]
    else:
        optical_flow1 = [None] + [None] + optical_flow1
        assert len(boxes) == len(scores) == len(reids) == len(classes) == len(masks) == len(optical_flow0)

    print("Creating detections_list and graph...")

    detections_list, G, _ = get_detections_G(boxes, scores, reids, classes, masks, class_to_track,
                                                       optical_flow0, optical_flow1, images, tracker_options)

    # cost_matrix0: detection cost matrix of the first order. Each (i,j)th element represents cost of going from detection i in frame t-1 to detection j in frame t
    # cost_matrix1: detection cost matrix of the second order. Each (i,j)th element represents cost of going from detection i in frame t-1 to detection j in frame t+1
    # lower cost is better.
    # a_node_cost: Nodes represent assignment nodes. If there are 20 best assignments for (t-1, t) frames, a_node_cost[t-1] will have 20 entries, representing total costs of those assignments.
    # a_node_assignment: The corresponding assignments whose costs are calculated in a_node_cost

    cost_matrix0 = []
    cost_matrix0_iou = []
    cost_matrix0_app = []

    cost_matrix1 = []
    cost_matrix1_iou = []
    cost_matrix1_app = []

    a_node_cost = []
    a_node_assignment = []


    ###################################################################################3

    print("Creating detection cost matrices, K-best assignments...")

    for t in range(0, len(detections_list)):

        if t > 0:
            cm, cm_iou, cm_app = get_cost_matrix0_i(detections_list, t, G, structured_mots_options)
            cost_matrix0.append(cm)
            cost_matrix0_iou.append(cm_iou)
            cost_matrix0_app.append(cm_app)

            # get 20 top assignments from each (t-1, t)
            temp_cost, temp_assignment = get_Kbest_assignments_per_frame_pair(cm, cm_iou, detections_list[t], structured_mots_options)
            a_node_cost.append(temp_cost)
            a_node_assignment.append(temp_assignment)

        if t > 1:
            cm1, cm1_iou, cm1_app = get_cost_matrix1_i(detections_list, t, G, structured_mots_options)
            cost_matrix1.append(cm1)
            cost_matrix1_iou.append(cm1_iou)
            cost_matrix1_app.append(cm1_app)

    ###################################################################################

    print("Creating assignment graph, assignment matrix...")

    a_cost_matrix0, G_assignment = get_assignment_cost_matrix_assignment_graph(a_node_assignment.copy(), a_node_cost,
                                                                     cost_matrix1, second_order=structured_mots_options[
            "second_order"], second_order_weight=structured_mots_options["second_order_weight"])

    #helper.show_graph_pyvis(G_assignment)

    print("Solving dp...")

    #optimized_paths = nx.dijkstra_path(G_assignment, "start", "end")
    #optimized_path_length = nx.dijkstra_path_length(G_assignment, "start", "end")
    a_cost_matrix0_np=[]
    for acm in a_cost_matrix0:
        a_cost_matrix0_np.append(acm.detach().cpu().numpy())
    assignment_list = dp.assignment_dyn_prog(a_cost_matrix0_np, a_node_assignment)
    pred_iou_cost = torch.FloatTensor([0.])
    pred_app_cost = torch.FloatTensor([0.])
    pred_cost = 0
    for ia in range(0, len(assignment_list)):
        assignment = assignment_list[ia][assignment_list[ia][:, 0] >= 0]
        assignment = assignment[assignment[:, 1] >= 0]
        for pairs in assignment:
            pred_cost += cost_matrix0[ia][pairs[0], pairs[1]]
            pred_iou_cost += cost_matrix0_iou[ia][pairs[0], pairs[1]]
            pred_app_cost += cost_matrix0_app[ia][pairs[0], pairs[1]]

    if training:
        y_star=assignment_list.copy()
        cost_gt, y_gt=get_gt_costmatrix(labels)
        mismatches, matches= calculate_mismatches(y_star, y_gt)

        track_loss_iou= sum(mismatches)+pred_iou_cost
        track_loss_app= sum(mismatches)+pred_app_cost#have to check this. lambda_app is not 1-lambda_iou anymore

        track_loss=torch.tensor([track_loss_iou, track_loss_app]).unsqueeze(0)

        return assignment_list, track_loss
    else:
        print("Saving Result...")

        result = [
            [track.track_id
             for track in tracks_t] for tracks_t in all_tracks]
        return result




    # For each iteration (n frame pairs):
    # L_obj: lambda [-C_gt - log sum over all possible assignments y (e^(C_y))]
    # Objective: Maximize L_obj w.r.t lambda
    # C_y: cost for assignment y in each frame pair (iou+appearance)
    # C_gt: cost for gt assignment in each frame pair (iou+appearance)

    # Gradient of L_obj w.r.t lambda:
    # L_obj_grad_lambda = sum over i (C_prime(y_i))[P_y_i - delta_function(y_i=y_gt)]
    # i: possible assignment
    # Making it a hard loss, P_y_i=1 if y_i=y_star_i, else 0
    # Adding a regularizer: R=0 for now
    # L_obj_grad_lambda_t   = C_prime(y_star_t) if y_i_t=y_star_t
    #                     = -C_prime(y_gt_t) if y_i_t=y_gt_t
    #                     = 0 otherwise
    # L_obj_grad_lambda = sum over t [(C_prime(y_star_t) - C_prime(y_gt_t)]
    # Introducing curly_L = C_prime

    # Objective: minimize w.r.t lambda: curly_L_star - curly_L_gt

    # When curly_L_gt = 0, we don't get any signal here about how good our y_star is, so we introduced Delta
    # Delta: sum(mismatches per frame)
    # Basically, Objective becomes: ((L_gt + Delta) - L_y_star) summing over n frames
    # Adding Regularizer R=0 for now

    # Final Objective: R + curly_L_gt for n frames + (Delta-curly_L_star) summing over n frames
    # Incorporating Delta in DP itself


    ###################################################################################

def get_detections_G(boxes, scores, reids, classes, masks, class_to_track, optical_flow0, optical_flow1, images,
                     tracker_options):
    detections_list = []
    G = nx.DiGraph()

    for t, (boxes_t, scores_t, reids_t, classes_t, masks_t, flow_tm0_t, flow_tm1_t, image) in enumerate(
            zip(boxes, scores, reids,
                classes, masks,
                optical_flow0,
                optical_flow1, images)):
        detections_t = []
        count = 0
        for box, score, reid, class_, msk in zip(boxes_t, scores_t, reids_t, classes_t, masks_t):
            if class_ != class_to_track:
                continue
            if msk.type()=='torch.cuda.FloatTensor':
                mask=cocomask.encode(np.asfortranarray(np.array(msk.detach().cpu()) > 0).astype(np.uint8))
            else:
                mask=msk.copy()
            if cocomask.area(mask) > 10 and score >= tracker_options["detection_confidence_threshold"]:
                detections_t.append((box, reid, msk, class_, score))
                G.add_node(str(t) + "_" + str(count))
                G.nodes[str(t) + "_" + str(count)]["detection"] = (box, reid, msk, class_, score)
                G.nodes[str(t) + "_" + str(count)]["reid"] = reid
                G.nodes[str(t) + "_" + str(count)]["node_cost"] = 1
                G.nodes[str(t) + "_" + str(count)]["image"] = image
                G.nodes[str(t) + "_" + str(count)]["flow0"] = flow_tm0_t
                G.nodes[str(t) + "_" + str(count)]["flow1"] = flow_tm1_t
                count += 1
            else:
                continue
        detections_list.append(detections_t)

    positions = helper.node_positions(G, False)
    return detections_list, G, positions



def get_Kbest_assignments_per_frame_pair(cm, cm_iou, det_t, structured_mots_options, aux_cost=1.):
    # det_t: detections in t
    # cm: detection cost matrix between t-1 and t
    # aux_cost: Cost of assignment to auxillary node
    K = structured_mots_options["K"]

    k = min(20, math.factorial(min(cm.shape[0], cm.shape[1])))
    cm_np=cm.detach().numpy()
    temp_cost, temp_assignment1 = k_best_costs(k, cm_np)
    temp_cost = temp_cost[:min(k, K)]
    temp_assignment1 = temp_assignment1[:min(k, K)]
    assignment1 = temp_assignment1.copy()

    temp_cost_tensor=[]
    for ta in temp_assignment1:
        pairs=ta[0]
        tc=cm[pairs[0],pairs[1]].clone()
        for pairs in ta[1:]:
            tc=tc+cm[pairs[0],pairs[1]]
        temp_cost_tensor.append(tc)
    print(temp_cost_tensor)
    for pos, temp_assignment in enumerate(temp_assignment1):

        assignment = temp_assignment.copy()
        assignment1[pos] = assignment

        diff = len(det_t) - len(assignment)
        # diff: no. of detections in t not assigned to detections in t-1.

        if diff>0:
            temp_cost_tensor[pos]= temp_cost_tensor[pos]+torch.tensor(diff*aux_cost) #############need to check the gradient here
            # Adding cost of assigning to auxillary node (aux_cost) for unassigned detections in t

    return temp_cost_tensor, assignment1

def get_assignment_cost_matrix_assignment_graph(node_assignment,node_cost,cost_matrix1, second_order=1,second_order_weight=1):
  # cost_matrix1: detection cost matrix of the 2nd order. Each (i,j)th element represents cost of going from detection i in frame t-1 to detection j in frame t+1
  # Note: We don't need cost_matrix0 (explained below) here, because node_cost already takes care of it
  # cost_matrix0: detection cost matrix of the 1nd order. Each (i,j)th element represents cost of going from detection i in frame t-1 to detection j in frame t

  # output: a_cost_matrix0. (i,j)th element in a_cost_matrix[t] represents cost of assigning a_i in (t-1) layer in a_j in (t) layer
  a_node_assignment=node_assignment.copy()
  a_node_cost=node_cost.copy()
  G = nx.DiGraph()

  a_cost_matrix0=[]

  #put -1 whenever there is no assignment, create graph nodes
  for i in range(0,len(a_node_cost)):

      for j in range(0,len(a_node_cost[i])):
          G.add_node(str(i)+"_"+str(j)) #level_j-th best
          G.nodes[str(i) + "_" + str(j)]["node_cost"] = a_node_cost[i][j].item()


          temp_assignment = a_node_assignment[i][j]
          #temp_cost = a_node_cost[i][j]
          if i<len(a_node_cost)-1:
              for k in range(0,cost_matrix1[i].shape[0]):
                  if k not in a_node_assignment[i][j][:,0]:
                      temp_assignment = np.concatenate(
                          [temp_assignment[:k], np.array([k, -1]).reshape(1, 2), temp_assignment[k:]])
                  #temp_cost +=1
              a_node_assignment[i][j]=temp_assignment
          G.nodes[str(i) + "_" + str(j)]["assignment"] = a_node_assignment[i][j]

      if len(a_node_cost[i])==0:
          G.nodes[str(i) + "_0"]["node_cost"] = 0
          G.nodes[str(i) + "_0"]["assignment"] = []


  for i in range(0, len(a_node_cost)):
      for j in range(0, len(a_node_cost[i])):
          temp_assignment = a_node_assignment[i][j]
          # temp_cost = a_node_cost[i][j]
          if i < len(a_node_cost) - 1:
            for k in range(0, cost_matrix1[i-1].shape[1]):
              if k not in a_node_assignment[i][j][:, 1]:
                  temp_assignment = np.concatenate(
                      [temp_assignment, np.array([-1,k]).reshape(1,2)])
                  # temp_cost +=1
            a_node_assignment[i][j] = temp_assignment
          G.nodes[str(i) + "_" + str(j)]["assignment"] = a_node_assignment[i][j]



  #create the assignment cost matrix, create graph edges

  for i in range(0, len(a_node_cost)-1):
      assignments_t1=a_node_assignment[i] #assignment at level1
      assignments_t2=a_node_assignment[i+1]


      a_cost_matrix=torch.zeros((len(assignments_t1),len(assignments_t2)))
      for j in range(0,len(assignments_t1)):
        for k in range(0,len(assignments_t2)):
          a_cost_matrix[j,k] += a_node_cost[i][j] + a_node_cost[i + 1][k]
          if second_order:

              second_order_term=find_second_order_term(assignments_t1, assignments_t2, j, k, cost_matrix1[i])
              total_term=a_cost_matrix[j, k]+second_order_weight*second_order_term
              G.add_edge(str(i) + "_" + str(j), str(i+1) + "_" + str(k), weight=total_term.detach().cpu().item())
              a_cost_matrix[j, k]=total_term

          else:
              G.add_edge(str(i) + "_" + str(j), str(i + 1) + "_" + str(k), weight=a_cost_matrix[j,k].detach().cpu().item())

      a_cost_matrix0.append(a_cost_matrix)
  if len(list(G.nodes))>0:
      start = list(G.edges)[0][0][:list(G.edges)[0][0].find("_")]
      stop = list(G.edges)[-1][1][:list(G.edges)[-1][1].find("_")] + "_0"
      G.add_node("start")
      G.add_node("end")
      stop=stop[:stop.find("_")]
      for n in list(G.nodes):
        if n[:n.find("_")]==start:
          G.add_edge("start",n,weight=1)
        if n[:n.find("_")]==stop:
          G.add_edge(n,"end",weight=1)

  return a_cost_matrix0, G

def find_second_order_term(assignments_t1,assignments_t2, j,k, cost_matrix1_i,threshold=0.9):
    second_term=0
    temp_t1 = assignments_t1[j][assignments_t1[j][:, 0] >= 0]
    temp_t2 = assignments_t2[k][assignments_t2[k][:, 1] >= 0]
    # if len(assignments_t1[j])<cost_matrix1[i].shape[0] and len(assignments_t2[k])<cost_matrix1[i].shape[1]:

    for l in range(0, len(temp_t1)):
        for m in range(0, len(temp_t2)):
            pair1 = temp_t1[l]
            pair2 = temp_t2[m]

            # print(pair1,pair2)
            if pair1[1] == pair2[0]:
                if pair1[1] != -1:
                    second_term += cost_matrix1_i[pair1[0], pair2[1]]
                elif cost_matrix1_i[pair1[0], pair2[1]]< threshold:

                    second_term+=cost_matrix1_i[pair1[0], pair2[1]]
                    #assignment_t1=


    return second_term


def get_cost_matrix0_i(detections_list, t, G, structured_mots_options):
    lambda_iou_main = structured_mots_options["lambda_iou_main"]
    lambda_app_main = structured_mots_options["lambda_app_main"]
    if len(detections_list[t]) > 0:
        if detections_list[t - 1] != []:
            flow_tm0_t = G.nodes[str(t) + "_" + str(0)]["flow0"]


            masks_t0, masks_tm0, masks_tm0_warped = helper.get_warped_mask(flow_tm0_t, detections_list[t],
                                                                    detections_list[t - 1])

            # show_masks(masks_tm0,masks_tm0_warped,masks_t0,dir_others+"mask/"+)
            cm_iou = np.transpose(
                1 - cocomask.iou(masks_t0, masks_tm0_warped, [False] * len(masks_tm0_warped)))  # cost_matrix0[i_cm]
            cm_app = helper.get_appearance_cm(G, detections_list, t - 1, t)
        else:
            c0 = len(detections_list[t])
            if c0 == 0:
                c0 = 1
            cm_iou = np.zeros((1, c0))
            cm_app = np.zeros((1, c0))
            # cm = np.zeros((1, c0))
    else:
        r0 = len(detections_list[t - 1])
        if r0 == 0:
            r0 = 1
        # cm=np.zeros((r0, 1))
        cm_iou = np.zeros((r0, 1))
        cm_app = np.zeros((r0, 1))

    cm_iou=torch.tensor(cm_iou)
    cm_app=torch.tensor(cm_app)
    cm = lambda_iou_main * cm_iou+ lambda_app_main * cm_app

    return cm, cm_iou, cm_app

def get_cost_matrix1_i(detections_list, t, G, structured_mots_options):
    lambda_iou_main = structured_mots_options["lambda_iou_main"]
    lambda_app_main = structured_mots_options["lambda_app_main"]
    if len(detections_list[t]) > 0:
        if detections_list[t - 2] != []:
            flow_tm1_t = G.nodes[str(t) + "_" + str(0)]["flow1"]
            masks_t1, masks_tm1, masks_tm1_warped = helper.get_warped_mask(flow_tm1_t, detections_list[t],
                                                                    detections_list[t - 2])
            # show_masks(masks_tm1, masks_tm1_warped, masks_t1, dir_others + "mask/")
            cm1_iou = np.transpose(1 - cocomask.iou(masks_t1, masks_tm1_warped, [False] * len(masks_tm1_warped)))
            cm1_app = helper.get_appearance_cm(G, detections_list, t - 2, t)
        else:
            c1 = len(detections_list[t])
            if c1 == 0:
                c1 = 1
            cm1_iou = np.zeros((1, c1))
            cm1_app = np.zeros((1, c1))
    else:
        r1 = len(detections_list[t - 2])
        if r1 == 0:
            r1 = 1
        cm1_iou = np.zeros((r1, 1))
        cm1_app = np.zeros((r1, 1))

    cm1_iou = torch.tensor(cm1_iou)
    cm1_app = torch.tensor(cm1_app)
    cm1 = lambda_iou_main * cm1_iou + lambda_app_main * cm1_app
    return cm1, cm1_iou, cm1_app

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/KITTI_MOTS/val/")
    parser.add_argument("--video", type=str, default="0002")  # path to imgs
    parser.add_argument("--detect_dir", type=str, default="../OUTPUTS/trackrcnn/detections_x152/val")
    parser.add_argument("--output_dir", type=str, default="../OUTPUTS/trackrcnn/tracking_x152_iou/val/Instances_txt")

    parser.add_argument("--lambda_iou_main", type=float, default=0.95)
    parser.add_argument("--lambda_app_leaf", type=float, default=0.8)
    parser.add_argument("--lambda_w2_leaf", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=20)

    parser.add_argument("--second_order", type=int, default=1)
    parser.add_argument("--second_order_weight", type=float, default=0.2)

    parser.add_argument("--leaf", type=int, default=0)

    return parser.parse_args()

def get_gt_costmatrix(gt_seq):

    cm_all=[]
    y_gt=[]
    for i in range(1,len(gt_seq)):
        cm = np.zeros((len(gt_seq[i-1]),len(gt_seq[i])))
        prev_ids=[it["track_id"] for it in gt_seq[i-1]]
        curr_ids = [it["track_id"] for it in gt_seq[i]]
        for j in range(0,len(prev_ids)):
            for k in range(0,len(curr_ids)):
                cm[j,k]=int((prev_ids[j]-curr_ids[k])!=0)# and curr_ids[k]!=-1)
                if cm[j,k]==0 and curr_ids[k]==-1:
                    cm[j, k] = 1
        y_temp=np.transpose(np.array(np.where(cm==0)))
        cm_all.append(cm)
        y_gt.append(y_temp)
    return cm_all, y_gt

def calculate_mismatches(y_star, y_gt):
    matches=[]
    mismatches=[]
    for t, (y_star_t, y_gt_t) in enumerate(zip(y_star, y_gt)):
        matches_t=0
        mismatches_t=0
        for ids in y_star_t:
            if ids[0] in y_gt_t[:, 0]:
                pos = np.where(y_gt_t[:, 0] == ids[0])[0][0]
                if y_star_t[np.where(y_star_t[:, 0] == ids[0])[0][0], 1] != y_gt_t[pos, 1]:
                    mismatches_t += 1
                else:
                    matches_t += 1

        mismatches.append(mismatches_t)
        matches.append(matches_t)
    return mismatches, matches

def my_collate(btch):
    batch=btch.copy()
    for i in batch:
        for j in i[1]:
            for l in j:
                if l["bbox"].dim()>1:
                    l["bbox"]=l["bbox"][0]
                    l["category_id"]=l["category_id"][0]
                    l["track_id"] = l["track_id"][0]

    # image_paths, gt_labels, vid, image_names
    image_paths = [item[0] for item in batch]
    gt_labels = [item[1] for item in batch]
    #print(gt_labels)

    vid = [item[2] for item in batch]
    image_names = [item[3] for item in batch]
    # target = [item[1] for item in batch]
    # target = torch.LongTensor(target)
    return (image_paths, gt_labels, vid, image_names)