import sys
import torch
import torch.nn as nn
import numpy as np
import os
import pdb
from skimage.segmentation import relabel_sequential
import pycocotools.mask as cocomask

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('PointTrack'))

from PointTrack.models import BranchedERFNet
from PointTrack.models.BranchedERFNet import TrackerOffsetEmb
from PointTrack.datasets.KittiMOTSDataset import *
from PointTrack.utils.utils import Cluster
from PointTrack.criterions.mots_seg_loss import MOTSSeg2Loss

cluster = Cluster()

class SE_model(nn.Module):
    def __init__(self, seg_model_path, reid_model_path, lambda_path=None, seg_device1=0, track_device=0, car=True, joint_train_load=False):
        super(SE_model, self).__init__()

        if car:
            foreground_weight=200
        else:
            foreground_weight=50

        self.seg_model = torch.nn.DataParallel(BranchedERFNet(num_classes=[4,1])).cuda(seg_device1)
        self.seg_loss = torch.nn.DataParallel(MOTSSeg2Loss(foreground_weight=foreground_weight, n_sigma=2)).cuda(seg_device1)

        self.reid_model = torch.nn.DataParallel(TrackerOffsetEmb(num_points=1000, margin=0.2,
                                                            env_points=500, border_ic=3, outputD=32,
                                                            category=True)).cuda(seg_device1)

        self.tracking_head = nn.Linear(8, 1, bias=False).cuda(track_device)

        if joint_train_load:
            self.load_state_dict(torch.load(seg_model_path), strict=True)
        else:
            state = torch.load(seg_model_path)
            self.seg_model.load_state_dict(state['model_state_dict'], strict=True)
            state = torch.load(reid_model_path)
            self.reid_model.load_state_dict(state['model_state_dict'], strict=True)
            del state
            if lambda_path!=None:
                state=torch.load(lambda_path)#model.tracking_head.weight[0]
                self.tracking_head.weight.data[0]=state
                del state

        if car:
            self.class_id=1
            self.dataset=MOTSTrackCarsValOffset(type="val",num_points=1500,gt=False, box=True,  category=True)
        if not car:
            self.class_id = 2 # pedestrian
            self.dataset=MOTSTrackPersonValOffset(type="val",num_points=1500,gt=False, box=True,  category=True)


    def forward(self, images, labels,Training=True, seg_device0=0, seg_device1=0, car=True):
        #seg_predictions=[]
        #seg_losses = []

        required_size=[384,1248]
        h,w =required_size[0]-images[0].shape[-2], required_size[1]-images[0].shape[-1]
        zero_pad=torch.nn.ZeroPad2d((0, w, 0, h))
        imgs=zero_pad(images[0].float()/255)

        if Training:
            op= self.seg_model(imgs)[:,:,:images[0].shape[2],:images[0].shape[3]]
        else:
            with torch.no_grad():
                op=self.seg_model(imgs)[:,:,:images[0].shape[2],:images[0].shape[3]]

        samples=[]
        for _,(img, output, label) in enumerate(zip(images[0].cpu().numpy(), op, labels)):


            instance_map = cluster.cluster_mots_wo_points(output, threshold=0.94,
                                                          min_pixel=160,
                                                          with_uv=True, n_sigma=2)
            sample=self.dataset.get_data_for_structured_mots(img.transpose(1,2,0)[:,:,[2,1,0]], instance_map.numpy())

            with torch.no_grad():
                sample["reids"] = self.get_reid(sample, device=seg_device1)
            #(output[4] + (1 - output[4].detach())) * torch.tensor(sample["masks"]).cuda(seg_device1)
            if sample["masks"]!=[]:
                sample["masks"]=(torch.stack([(i+(1-i.detach())) for i in output]).sum(0)/5)\
                            *torch.tensor(sample["masks"]).cuda(seg_device1)
            else:
                sample["masks"]=torch.tensor(sample["masks"]).cuda(seg_device1)
            samples.append(sample)

        instances, class_labels = self.decode_instances(labels, self.class_id, img_shape=images[0][0][0].shape)
        if Training:
            loss = self.seg_loss(op, instances, class_labels,iou=True)
        else:
            with torch.no_grad():
                loss = self.seg_loss(op, instances, class_labels, iou=True)

        loss = loss.mean()

        boxes, scores, masks, reids, classes= self.format_predictions(samples, car=car)


        return boxes, scores, masks, reids, classes, loss

    def get_reid(self,sample, device=0):
        points = sample['points']
        if len(points) < 1:
            reids = np.array([])
        else:
            xyxys = sample['xyxys']
            reids = self.reid_model(torch.tensor([points]), None, torch.tensor([xyxys]).cuda(device), infer=True)

        return reids

    def format_predictions(self, samples, device=0, car=True):
        if car:
            class_id = 1
        else:
            class_id = 2
        masks = [torch.stack([m for m in sample["masks"]])
                 if [m for m in sample["masks"]] != [] else torch.tensor([]) for sample in samples]
        reids = [torch.stack([r.cuda(device) for r in sample["reids"]])
                 if [m for m in sample["masks"]] != [] else torch.tensor([]) for sample in samples]

        scores = [torch.stack([torch.tensor(1).cuda(device) for m in sample["masks"]])
                  if [m for m in sample["masks"]] != [] else torch.tensor([]) for sample in samples]
        classes = [torch.stack([torch.tensor(class_id).cuda(device) for m in sample["masks"]])
                   if [m for m in sample["masks"]] != [] else torch.tensor([]) for sample in samples]

        boxes = [torch.stack([torch.tensor(
            cocomask.toBbox(cocomask.encode(np.asfortranarray(np.array(m.detach().cpu()).astype(np.uint8))))).cuda(
            device)
                              for m in sample["masks"]])
                 if [m for m in sample["masks"]] != [] else torch.tensor([]) for sample in samples]

        for b in range(len(boxes)):
            if list(boxes[b]) != []:
                boxes[b][:, 2] = boxes[b][:, 2] + boxes[b][:, 0]
                boxes[b][:, 3] = boxes[b][:, 3] + boxes[b][:, 1]

        return boxes, scores, masks, reids, classes

    def decode_instances(self, labels, class_id, img_shape=[5,5]): #get semantic map and instance map
        instances=[]
        class_maps=[]
        for label_t in labels:
            pic=np.zeros(np.array(img_shape), dtype=np.uint8)
            for i,lb in enumerate(label_t):
                pic[np.where(np.array(lb["segmentation"].squeeze(0)))]=i+1


            instance_map = np.zeros_like(pic)
                #(pic.shape[0], pic.shape[1]), dtype=np.uint8)

            # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
            class_map = np.zeros_like(pic)
                #(pic.shape[0], pic.shape[1]), dtype=np.uint8)

            mask=np.zeros_like(pic)
            for lb in label_t:
                if lb["category_id"]==class_id-1:
                    mask=mask+np.array(lb["segmentation"].squeeze(0))

            mask = mask >0
            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(pic[mask])
                instance_map[mask] = ids
                class_map[mask] = 1

            instances.append(instance_map)
            class_maps.append(class_map)

        return torch.tensor(np.stack(instances)), torch.tensor(np.stack(class_maps))

