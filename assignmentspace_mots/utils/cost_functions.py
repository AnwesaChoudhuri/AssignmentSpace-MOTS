import torch
from cv2 import remap, INTER_NEAREST
import numpy as np
import cv2
import sys
from math import ceil
import os
#import pycocotools.mask as cocomask
import pdb
def warp_displays(mask_copy, res,flow,i):
    if not os.path.exists(str(i)):
        os.mkdir(str(i))
    flow_img=cv2.imread("data/KITTI_MOTS/train/flow_skip0_img/0001/000014.png")#255-optflow_computeImg(flow)#
    mask_dup=np.expand_dims(mask_copy, 2)
    mask_temp=np.concatenate([mask_dup, mask_dup, mask_dup],axis=2)
    cv2.imwrite(str(i)+"/mask_temp.png", mask_temp * flow_img)

    fctr=int((mask_copy.sum()**(1/2))/3)
    kernel = np.ones((fctr, fctr), np.uint8)
    mask_erode = cv2.erode(mask_copy, kernel, iterations=1)
    mask_dup2 = np.expand_dims(np.array(mask_erode), 2)
    mask_temp2 = np.concatenate([mask_dup2, mask_dup2, mask_dup2], axis=2)
    cv2.imwrite(str(i)+"/mask_temp_erode.png", mask_temp2 * flow_img)
    cv2.imwrite(str(i)+"/mask_erode_diff.png", (mask_temp-mask_temp2) * 255)

     #earlier mask_copy instead of mask_erode
    res2 = np.expand_dims(res, 2)
    res_temp = np.concatenate([res2, res2, res2], axis=2)
    cv2.imwrite(str(i)+"/mask_warped.png", (res_temp)* flow_img)


    res_erode = remap(mask_erode, flow, None, INTER_NEAREST) #earlier mask_copy instead of mask_erode
    res_erode2 = np.expand_dims(res_erode, 2)
    res_erode_temp = np.concatenate([res_erode2, res_erode2, res_erode2], axis=2)
    cv2.imwrite(str(i)+"/mask_erode_warped.png", (res_erode_temp) * flow_img)

    res_dilate = cv2.dilate(res_erode, kernel, iterations=1)
    res_dilate2 = np.expand_dims(res_dilate, 2)
    res_dilate_temp = np.concatenate([res_dilate2, res_dilate2, res_dilate2], axis=2)
    cv2.imwrite(str(i)+"/mask_dilate_warped.png", (res_dilate_temp) * flow_img)
    return res_dilate

def warp_flow(mask, flow, device=0,i=0):
    # warp
    #print(i)

    # erode the boundaries

    mask_copy=np.array(mask.detach().cpu())
    res = remap(mask_copy, flow, None, INTER_NEAREST)
    #res_erode_dilate = warp_displays(mask_copy, res,flow,i)


    #warped = Variable(torch.tensor(np.equal(res, 1).astype(np.float64)), requires_grad=True)
    mask.data=torch.tensor(np.equal(res, 1).astype(np.float64)).cuda(device)
    return mask

def get_warped_mask(flow_tm_t, masks_tm, device=0): #dont use this one for training. parallel code written next

    if flow_tm_t is not None:
        h, w = flow_tm_t.shape[:2]
        #flow_tm_t = -flow_tm_t # already -ve optical flow is inputted
        flow_tm_t[:, :, 0] += np.arange(w)
        flow_tm_t[:, :, 1] += np.arange(h)[:, np.newaxis]

        masks_tm_warped = torch.stack([warp_flow(mask, flow_tm_t, device=device) for mask in masks_tm])

    else:
        masks_tm_warped = masks_tm

    return masks_tm_warped

def get_warped_mask_parallel(flow_tm_ts, masks_tm, device=0):
    ct=0
    masks_tm_warped=[]
    for _, (masks_, flow_tm_t) in enumerate(zip(masks_tm, flow_tm_ts)):
        if flow_tm_t is not None and len(masks_)>0:
            h, w = flow_tm_t.shape[:2]
            #flow_tm_t = -flow_tm_t # already -ve optical flow is inputted
            flow_tm_t[:, :, 0] += np.arange(w)
            flow_tm_t[:, :, 1] += np.arange(h)[:, np.newaxis]
            masks_tm_warped.append(torch.stack([torch.stack([warp_flow(m, flow_tm_t, device,i)  for m in mask]) for i, mask in enumerate(masks_)]))

        else:
            masks_tm_warped.append(masks_)
        ct=ct+1

    return torch.stack(masks_tm_warped)

    # flow_torch=[]
    # for fl in range(masks_tm.shape[0]):
    #     for fl2 in range(masks_tm.shape[1]*masks_tm.shape[2]):
    #         flow_torch.append(torch.tensor(flow_tm_ts[fl]))
    #
    # masks_tm_warped=F.grid_sample(
    #     masks_tm.reshape(masks_tm.shape[0]*masks_tm.shape[1]*masks_tm.shape[2], 1, masks_tm.shape[3], masks_tm.shape[4]),
    #     torch.stack(flow_torch).cuda(device), align_corners=True).reshape(masks_tm.shape)

    #return masks_tm_warped

def find_cost_matrix_bb_dist(det_boxes,det_next_boxes, scale=[375,1242]):
    x_m1, y_m1 = (det_boxes[:,0] + det_boxes[:,2]) / 2, (det_boxes[:,1] + det_boxes[:,3]) / 2
    x_m2, y_m2 = (det_next_boxes[:,0] + det_next_boxes[:,2]) / 2, (det_next_boxes[:,1] + det_next_boxes[:,3]) / 2
    cost_x = (x_m1.unsqueeze(1) - x_m2.unsqueeze(0))
    cost_y = (y_m1.unsqueeze(1) - y_m2.unsqueeze(0))
    cost_dist = (((cost_x ** 2 + cost_y ** 2) ** (1 / 2)))/((scale[0]**2+scale[1]**2)**(1/2))

    return cost_dist

def find_cost_matrix_bb_dist_parallel(det_boxes,det_next_boxes, scale=[375,1242]):

    x_m1, y_m1 = (det_boxes[:,:,:,0] + det_boxes[:,:,:,2]) / 2, (det_boxes[:,:,:,1] + det_boxes[:,:,:,3]) / 2
    x_m2, y_m2 = (det_next_boxes[:,:,:,0] + det_next_boxes[:,:,:,2]) / 2, (det_next_boxes[:,:,:,1] + det_next_boxes[:,:,:,3]) / 2
    cost_x = (x_m1 - x_m2)
    cost_y = (y_m1 - y_m2)
    cost_dist = ((cost_x ** 2 + cost_y ** 2) ** (1 / 2))/((scale[0]**2+scale[1]**2)**(1/2))

    return cost_dist

#cos_sim = nn.CosineSimilarity(dim=3, eps=1e-6)
def find_cost_matrix_app_parallel(det_app, det_next_app, mots=True):

    if mots:
        cost_app = torch.cdist(det_app.squeeze(2),det_next_app.squeeze(1))
    else:
        cost_app = torch.mm(det_app.squeeze(2),det_next_app.squeeze(1))
    #cost_app=cos_sim(det_app, det_next_app)

    return 1-cost_app

def find_cost_matrix_app(det_app, det_next_app, mots=True):

    #cost_app = cos_sim(torch.stack(det_app).unsqueeze(1).unsqueeze(0),torch.stack(det_next_app).unsqueeze(0).unsqueeze(0)).squeeze(0)
    #cost_app = cos_sim(torch.stack(det_app).unsqueeze(1).unsqueeze(0),
                      # torch.stack(det_next_app).unsqueeze(0).unsqueeze(0)).squeeze(0)

    if mots:
        cost_app = torch.cdist(torch.stack(det_app).unsqueeze(0),
                       torch.stack(det_next_app).unsqueeze(0)).squeeze(0)
    else:
        cost_app = torch.mm(torch.stack(det_app),torch.stack(det_next_app).permute(1,0))

    return 1-cost_app

def find_cost_matrix_iou(det, det_next, iou="bb", mem_limit=10, optflow=[],parallel=0, device=1):
    #parallel=0 is for serial computation of each mask_iou for each pair of detections in frame t and t+1


    if iou == "mask":

        #print(det_next["masks"])
        mask_dt_next = (torch.stack(det_next["masks"]) + (1 - torch.stack(det_next["masks"])).detach()) * (torch.stack(det_next["masks"]).detach() > 0.5).unsqueeze(0)


        del det_next
        #mask_dt = ((torch.stack(det["masks"]) + (1 - torch.stack(det["masks"])).detach())* (torch.stack(det["masks"]) > 0.5)).unsqueeze(1)
        if len(optflow) == 0:

            mask_dt = ((torch.stack(det["masks"]) + (1 - torch.stack(det["masks"])).detach()) * (
                        torch.stack(det["masks"]).detach() > 0.5)).unsqueeze(1)
            del det
        else:

            mask_dt_temp = ((torch.stack(det["masks"]) + (1 - torch.stack(det["masks"])).detach()) * (torch.stack(det["masks"]).detach() > 0.5))
            mask_dt = get_warped_mask(optflow, mask_dt_temp, device=device).unsqueeze(1)
            #del mask_dt_temp, det, optflow



            #check
        if mask_dt.shape[0]>mem_limit and mask_dt_next.shape[1]<=mem_limit:
            mask_dt1=mask_dt[:int(mask_dt.shape[0]/2)]
            mask_dt2 = mask_dt[int(mask_dt.shape[0]/2): ]
            del mask_dt
            cost_iou1 = (mask_dt1 * mask_dt_next).float().sum(2).sum(2) / (mask_dt1 + mask_dt_next).sum(2).sum(2)
            cost_iou2 = (mask_dt2 * mask_dt_next).float().sum(2).sum(2) / (mask_dt2 + mask_dt_next).sum(2).sum(2)
            cost_iou = torch.cat([cost_iou1, cost_iou2])
            del cost_iou1
            del cost_iou2
            del mask_dt1
            del mask_dt2
            del mask_dt_next
        elif mask_dt_next.shape[1]>mem_limit and mask_dt.shape[0]<=mem_limit:
            mask_dt_next1=mask_dt_next[:,:int(mask_dt_next.shape[1]/2)]
            mask_dt_next2 = mask_dt_next[:,int(mask_dt_next.shape[1]/2): ]
            del mask_dt_next
            cost_iou1 = (mask_dt * mask_dt_next1).float().sum(2).sum(2) / (mask_dt+ mask_dt_next1).sum(2).sum(2)
            cost_iou2 = (mask_dt * mask_dt_next2).float().sum(2).sum(2) / (mask_dt + mask_dt_next2).sum(2).sum(2)
            cost_iou = torch.cat([cost_iou1, cost_iou2], dim=1)
            del cost_iou1
            del cost_iou2
            del mask_dt_next1
            del mask_dt_next2
            del mask_dt
        elif mask_dt.shape[0]>mem_limit and mask_dt_next.shape[1]>mem_limit and parallel==1:
            mask_dt1=mask_dt[:int(mask_dt.shape[0]/2)]
            mask_dt2 = mask_dt[int(mask_dt.shape[0]/2): ]
            mask_dt_next1 = mask_dt_next[:, :int(mask_dt_next.shape[1] / 2)]
            mask_dt_next2 = mask_dt_next[:, int(mask_dt_next.shape[1] / 2):]

            del mask_dt
            del mask_dt_next



            cost_iou1 = (mask_dt1 * mask_dt_next1).float().sum(2).sum(2) / (mask_dt1 + mask_dt_next1).sum(2).sum(2)
            cost_iou2 = (mask_dt2 * mask_dt_next1).float().sum(2).sum(2) / (mask_dt2 + mask_dt_next1).sum(2).sum(2)
            cost_iou3 = (mask_dt1 * mask_dt_next2).float().sum(2).sum(2) / (mask_dt1 + mask_dt_next2).sum(2).sum(2)
            cost_iou4 = (mask_dt2 * mask_dt_next2).float().sum(2).sum(2) / (mask_dt2 + mask_dt_next2).sum(2).sum(2)

            del mask_dt1, mask_dt2, mask_dt_next1, mask_dt_next2

            cost_iou = torch.cat([torch.cat([cost_iou1, cost_iou2]),torch.cat([cost_iou3, cost_iou4])], dim=1)
            del cost_iou1, cost_iou2, cost_iou3, cost_iou4
        elif mask_dt.shape[0] > mem_limit and mask_dt_next.shape[1] > mem_limit and parallel == 0:
            cost_iou = torch.zeros(mask_dt.shape[0],mask_dt_next.shape[1],dtype=torch.float64).cuda(device)
            # cost_grad_lmbda_matrix = torch.zeros((len(det), len(det_next), 2))
            # cost_grad_theta_matrix = torch.zeros((len(det), len(det_next), theta.shape[0], theta.shape[1], theta.shape[2]))
            for i1 in range(mask_dt.shape[0]):

                for i2 in range(mask_dt_next.shape[1]):

                    cost_iou[i1, i2] = (mask_dt[i1,0] * mask_dt_next[0,i2]).float().sum() / (mask_dt[i1,0] + mask_dt_next[0,i2]).sum()
                    #print("i am here")

        else:

            cost_iou = (mask_dt * mask_dt_next).float().sum(2).sum(2) / (mask_dt + mask_dt_next).sum(2).sum(2) # mask iou
            del mask_dt
            del mask_dt_next

    if iou == "bb":

        det_boxes=torch.stack(det["boxes"])
        det_next_boxes=torch.stack(det_next["boxes"])
        zeros=torch.zeros((len(det_boxes),len(det_next_boxes))).cuda(device)
        common_x = torch.max(zeros, torch.min((det_boxes[:,2].unsqueeze(1) - det_next_boxes[:,0].unsqueeze(0)), (det_next_boxes[:,2].unsqueeze(0) - det_boxes[:,0].unsqueeze(1))))
        common_y = torch.max(zeros, torch.min((det_boxes[:,3].unsqueeze(1) - det_next_boxes[:,1].unsqueeze(0)), (det_next_boxes[:,3].unsqueeze(0) - det_boxes[:,1].unsqueeze(1))))

        intersection = (common_x * common_y)

        union = ((det_boxes[:,2] - det_boxes[:,0]) * (det_boxes[:,3] - det_boxes[:,1])).unsqueeze(1) + \
                ((det_next_boxes[:,2] - det_next_boxes[:,0]) * (det_next_boxes[:,3] - det_next_boxes[:,1])).unsqueeze(0) - intersection
        cost_iou = intersection / union  # bb iou

    return 1-cost_iou

def find_cost_matrix_iou_parallel(det, det_next, iou="bb", mem_limit=30, optflows=[], device=1):


    if iou == "mask":

        #print(det_next["masks"])
        mask_dt_next = (det_next["masks"] + (1 - det_next["masks"].detach())) * (det_next["masks"].detach() > 0.5)

        del det_next
        #mask_dt = ((torch.stack(det["masks"]) + (1 - torch.stack(det["masks"])).detach())* (torch.stack(det["masks"]) > 0.5)).unsqueeze(1)
        if len(optflows[0]) == 0:
            mask_dt = ((det["masks"] + (1 - det["masks"].detach())) * (
                        det["masks"].detach() > 0.5))
            #del det
        else:
            mask_dt_temp = ((det["masks"] + (1 - det["masks"].detach())) * (
                    det["masks"].detach() > 0.5))
            mask_dt = get_warped_mask_parallel(optflows, mask_dt_temp, device=device)
            #del mask_dt_temp, det, optflows
        div=(mask_dt + mask_dt_next).sum(3).sum(3)
        if 0 in div:
            div[torch.where(div==0)]=div[torch.where(div==0)]+0.01

        cost_iou = ((mask_dt * mask_dt_next).float().sum(3).sum(3) / div)
             # mask iou
        #del mask_dt
        #del mask_dt_next

    if iou == "bb":
        det_boxes = det["boxes"]
        det_next_boxes = det_next["boxes"]
        zeros = torch.zeros(len(det_boxes), det_boxes.shape[1], det_next_boxes.shape[2]).cuda(device)
        common_x = torch.max(zeros, torch.min((det_boxes[:, :, :, 2] - det_next_boxes[:, :, :, 0]),
                                              (det_next_boxes[:, :, :, 2] - det_boxes[:, :, :, 0])))
        common_y = torch.max(zeros, torch.min((det_boxes[:, :, :, 3] - det_next_boxes[:, :, :, 1]),
                                              (det_next_boxes[:, :, :, 3] - det_boxes[:, :, :, 1])))

        intersection = (common_x * common_y)

        union = ((det_boxes[:, :, :, 2] - det_boxes[:, :, :, 0]) * (det_boxes[:, :, :, 3] - det_boxes[:, :, :, 1])) + (
                    (det_next_boxes[:, :, :, 2] - det_next_boxes[:, :, :, 0]) * (
                        det_next_boxes[:, :, :, 3] - det_next_boxes[:, :, :, 1])) - intersection
        union[torch.where(union==0)]=0.001
        cost_iou = intersection / union

    return 1-cost_iou


def find_cost_matrix(args, det, det_next, lmbda, size, optflow=[], parallel=False, img_shape=[375, 1242], reid_shape=50176):
    if args.real and len(det["boxes"])>0 and len(det_next["boxes"])>0:

        if parallel:

            max_boxes_det=np.array([len(boxes) for boxes in det["boxes"]]).max()
            det_boxes_tensor=torch.zeros(len(det["boxes"]), max_boxes_det, 1, 4).cuda(args.track_device)
            det_masks_tensor = torch.zeros(len(det["boxes"]), max_boxes_det, 1,img_shape[0], img_shape[1]).cuda(args.track_device)
            det_app_tensor = torch.zeros(len(det["boxes"]), max_boxes_det, 1, reid_shape).cuda(args.track_device)

            for t in range(len(det["boxes"])):
                if len(det["boxes"][t])>0:
                    det_boxes_tensor[t, :len(det["boxes"][t]),0]=torch.stack(det["boxes"][t])
                    det_masks_tensor[t, :len(det["boxes"][t]), 0] = torch.stack(det["masks"][t])
                    det_app_tensor[t, :len(det["boxes"][t]), 0] = torch.stack(det["reids"][t])

            max_boxes_det_next = np.array([len(boxes) for boxes in det_next["boxes"]]).max()
            det_next_boxes_tensor = torch.zeros(len(det_next["boxes"]), 1, max_boxes_det_next, 4).cuda(args.track_device)
            det_next_masks_tensor = torch.zeros(len(det_next["boxes"]), 1, max_boxes_det_next, img_shape[0], img_shape[1]).cuda(args.track_device)
            det_next_app_tensor = torch.zeros(len(det_next["boxes"]), 1, max_boxes_det_next, reid_shape).cuda(args.track_device)
            #print(det_masks_tensor.shape, det_next_masks_tensor.shape)

            for t in range(len(det_next["boxes"])):
                if len(det_next["boxes"][t]) > 0:
                    det_next_boxes_tensor[t, 0, :len(det_next["boxes"][t])] = torch.stack(det_next["boxes"][t])
                    det_next_masks_tensor[t, 0, :len(det_next["boxes"][t])] = torch.stack(det_next["masks"][t])
                    det_next_app_tensor[t, 0, :len(det_next["boxes"][t])] = torch.stack(det_next["reids"][t])

            cost_dist = find_cost_matrix_bb_dist_parallel(det_boxes_tensor, det_next_boxes_tensor)
            #lower is better

            cost_iou = find_cost_matrix_iou_parallel({"masks":det_masks_tensor, "boxes":det_boxes_tensor},
                                                     {"masks":det_next_masks_tensor, "boxes": det_next_boxes_tensor},
                                                     iou=args.iou_type, optflows=optflow, device=args.track_device)
            # lower is better

            cost_app = find_cost_matrix_app_parallel(det_app_tensor, det_next_app_tensor)

        else:
            cost_dist = find_cost_matrix_bb_dist(torch.stack(det["boxes"]),torch.stack(det_next["boxes"]))

            # to check if it matches with the parallel counterpart:
            # det_boxes_tensor=torch.stack(det["boxes"]).unsqueeze(0).unsqueeze(2)
            # det_next_boxes_tensor=torch.stack(det_next["boxes"]).unsqueeze(0).unsqueeze(1)
            # cost_dist_p = find_cost_matrix_bb_dist_parallel(det_boxes_tensor, det_next_boxes_tensor)

            cost_iou = find_cost_matrix_iou(det, det_next, iou=args.iou_type,optflow=optflow.copy(), device=args.track_device)
            #lower is better

            # to check if it matches with the parallel counterpart:
            # det_masks_tensor=torch.stack(det["masks"]).unsqueeze(0).unsqueeze(2)
            # det_next_masks_tensor=torch.stack(det_next["masks"]).unsqueeze(0).unsqueeze(1)
            # cost_iou_p = find_cost_matrix_iou_parallel({"masks":det_masks_tensor, "boxes":det_boxes_tensor},{"masks":det_next_masks_tensor, "boxes": det_next_boxes_tensor},iou=iou, optflows=[optflow.copy()], device=device)


            cost_app = find_cost_matrix_app(det["reids"], det_next["reids"],mots=args.mots)

            # to check if it matches with the parallel counterpart:
            # det_app_tensor=torch.stack(det["reids"]).unsqueeze(0).unsqueeze(2)
            # det_next_app_tensor=torch.stack(det_next["reids"]).unsqueeze(0).unsqueeze(1)
            # cost_app_p = find_cost_matrix_bb_dist_parallel(det_app_tensor, det_next_app_tensor)



        # note: there will be nan values in all the cost matrices and that's okay. will remove them later

        cost_matrix = lmbda[0] * cost_dist + lmbda[1] * cost_iou + lmbda[2] * cost_app + lmbda[3]


    else:

        cost_matrix = torch.zeros((len(det["boxes"]), len(det_next["boxes"]))).cuda(args.track_device)
        cost_iou = torch.zeros((len(det["boxes"]), len(det_next["boxes"]))).cuda(args.track_device)
        #cost_grad_lmbda_matrix = torch.zeros((len(det), len(det_next), 2))
        #cost_grad_theta_matrix = torch.zeros((len(det), len(det_next), theta.shape[0], theta.shape[1], theta.shape[2]))
        for i1 in range(len(det["boxes"])):
            dt=det["boxes"][i1]

            for i2 in range(len(det_next["boxes"])):
                dt_next = det_next["boxes"][i2]
                # print(i1, i2, dt, dt_next, find_cost(dt, dt_next, lmbda, size=size)[0])
                if args.real:
                    cost_matrix[i1, i2] = find_cost_real({"box":dt, "mask": det["masks"][i1]},
                                                         {"box":dt_next, "mask": det_next["masks"][i2]}, lmbda, size=size, quadratic=args.quadratic, iou="bb")
                else:
                    cost_matrix[i1, i2] = find_cost_simple(dt, dt_next,lmbda, size=size, quadratic=args.quadratic)
    return cost_matrix, cost_iou, []#,cost_clr

def find_cost_simple(dt, dt_next, lmbda, size=[10,10], quadratic=0):
    x_m1, y_m1 = (dt[0]+dt[2])/2, (dt[1]+dt[3])/2
    x_m2, y_m2 = (dt_next[0]+dt_next[2])/2, (dt_next[1]+dt_next[3])/2

    if x_m1 - x_m2 > 0:
        cost_x = x_m1 - x_m2

    elif x_m2 - x_m1 >= 0:
        cost_x = x_m2 - x_m1

    if y_m1 - y_m2 > 0:
        cost_y = y_m1 - y_m2

    elif y_m2 - y_m1 >= 0:
        cost_y = y_m2 - y_m1


    cost = lmbda[0] * cost_x**2 + lmbda[1] * cost_y**2

    if quadratic==1:
        cost = lmbda[0] * cost_x ** 2 + lmbda[1] * cost_y ** 2

    else:
        cost = lmbda[0] * cost_x ** 2 + lmbda[1] * cost_y ** 2
    if cost == 0.0:
        cost = 0.001

    return cost

def find_cost_real(dt, dt_next, lmbda, size=[10,10],quadratic=0, iou="bb"): #iou can be "bb" or "mask"

    x_m1, y_m1 = (dt["box"][0]+dt["box"][2])/2, (dt["box"][1]+dt["box"][3])/2
    x_m2, y_m2 = (dt_next["box"][0]+dt_next["box"][2])/2, (dt_next["box"][1]+dt_next["box"][3])/2
    cost_x = (x_m1 - x_m2)
    cost_y = (y_m1 - y_m2)
    cost_dist=(cost_x**2+cost_y**2)**(1/2)

    if iou=="mask":
        mask_dt=(dt["mask"]+(1-dt["mask"]).detach())*(dt["mask"]>0.5)
        mask_dt_next = (dt_next["mask"] + (1 - dt_next["mask"]).detach()) * (dt_next["mask"] > 0.5)
        cost_iou=(mask_dt*mask_dt_next).float().sum()/(mask_dt+mask_dt_next).sum() # mask iou


    if iou=="bb":
        common_x = max(0, min((dt["box"][2] - dt_next["box"][0]), (dt_next["box"][2] - dt["box"][0])))
        common_y = max(0, min((dt["box"][3] - dt_next["box"][1]), (dt_next["box"][3] - dt["box"][1])))

        intersection=(common_x*common_y)

        union= ((dt["box"][2]-dt["box"][0])*(dt["box"][3]-dt["box"][1]))+\
           ((dt_next["box"][2]-dt_next["box"][0])*(dt_next["box"][3]-dt_next["box"][1]))- intersection
        cost_iou=intersection/union # bb iou


    if quadratic == 1:
        cost = lmbda[0] * cost_dist**2 + lmbda[1] * cost_iou**2

    else:
        cost = lmbda[0] * cost_dist + lmbda[1] * cost_iou


    if cost == 0.0:
        cost = 0.001


    return cost

def find_cost(dt, dt_next, lmbda, size=[10,10]):
    mask1 = np.zeros((size))
    x1_m1, x2_m1 = [int(max(0, dt[0])), int(min(size[0] - 1, dt[2]))]
    y1_m1, y2_m1 = [int(max(0, dt[1])), int(min(size[1] - 1, dt[3]))]
    mask1[x1_m1:x2_m1, y1_m1:y2_m1] = 1

    mask2 = np.zeros((size))
    x1_m2, x2_m2 = [int(max(0, dt_next[0])), int(min(size[0] - 1, dt_next[2]))]
    y1_m2, y2_m2 = [int(max(0, dt_next[1])), int(min(size[1] - 1, dt_next[3]))]
    mask2[x1_m2:x2_m2, y1_m2:y2_m2] = 1
    if ((mask1 + mask2) > 0).sum():
        cost_iou = 1 - (mask1 * mask2).float().sum() / ((mask1 + mask2) > 0).sum()
        overlap_x=x2_m1-2*x1_m1+2*x2_m2-x1_m2
        overlap_y = y2_m1 - 2 * y1_m1 + 2 * y2_m2 - y1_m2

    else:
        cost_iou=1

    x_m1=(x1_m1+x2_m1)/2
    y_m1 = (y1_m1 + y2_m1) / 2

    x_m2 = (x1_m2 + x2_m2) / 2
    y_m2 = (y1_m2 + y2_m2) / 2

    distance=((x_m1-x_m2)**2+(y_m1-y_m2)**2)**(1/2)

    cost=lmbda[0]*cost_iou+lmbda[1]*distance
    cost_grad_lmbda=np.array([cost_iou, distance])
    cost_grad_theta = np.array([cost_iou, distance])

    if cost==0.0:
        cost=0.001

    return cost, cost_grad_lmbda


def makeColorwheel():

    #  color encoding scheme

    #   adapted from the color circle idea described at
    #   http://members.shaw.ca/quadibloc/other/colint.htm

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])  # r g b

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    col += RY

    # YG
    colorwheel[col:YG + col, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
    colorwheel[col:YG + col, 1] = 255;
    col += YG;

    # GC
    colorwheel[col:GC + col, 1] = 255
    colorwheel[col:GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
    col += GC;

    # CB
    colorwheel[col:CB + col, 1] = 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
    colorwheel[col:CB + col, 2] = 255
    col += CB;

    # BM
    colorwheel[col:BM + col, 2] = 255
    colorwheel[col:BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
    col += BM;

    # MR
    colorwheel[col:MR + col, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
    colorwheel[col:MR + col, 0] = 255
    return colorwheel

def computeColor(u, v):

    colorwheel = makeColorwheel();
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v)

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)  # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)  # 1, 2, ..., ncols
    k1 = k0 + 1;
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1], 3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
        col[~idx] *= 0.75  # out of range
        img[:, :, 2 - i] = np.floor(255 * col).astype(np.uint8)

    return img.astype(np.uint8)

def optflow_computeImg(flow):

    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1
    # fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0
    v[greater_v] = 0

    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])

    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    maxrad = max([maxrad, np.amax(rad)])
    # print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

    u = u / (maxrad + eps)
    v = v / (maxrad + eps)
    img = computeColor(u, v)
    return img
