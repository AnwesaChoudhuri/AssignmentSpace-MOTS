import sys
sys.path.append("../../detectron2/")
import logging
import numpy as np
import os
from collections import OrderedDict
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
import torch.nn as nn
import pycocotools.mask as cocomask
from detectron2.utils.events import EventStorage
import structured_mots_for_training as smt
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
import cv2
from detectron2.engine import default_argument_parser, default_setup, launch
from torch.nn import functional as F
from detectron2.modeling import build_model # in the file detectron2/modeling/meta_Arch/build.py
from tools.plain_train_net import setup
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances, Boxes, BoxMode, BitMasks
from detectron2 import model_zoo
import pdb
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference, mask_rcnn_loss
from detectron2.modeling.roi_heads import select_foreground_proposals
sys.path.append("../../PointTrack_undisturbed/")
from models.BranchedERFNet import TrackerOffsetEmb
from datasets.KittiMOTSDataset import *

class MASKRCNN_model(nn.Module):
    def __init__(self, existing_cfg, my_cfg,restore=False, mrcnn_device0=0, mrcnn_device1=0, track_device=0):
        super(MOTS_model, self).__init__()
        if not restore:
            existing_model = build_model(existing_cfg)
            DetectionCheckpointer(existing_model).load(existing_cfg.MODEL.WEIGHTS)
        my_model = build_model(my_cfg)
        DetectionCheckpointer(my_model).load(my_cfg.MODEL.WEIGHTS)
        if not restore:

            state_dict = my_model.roi_heads.state_dict()
            if "box_predictor.bbox_pred.weight" in state_dict.keys():
                add_ = ["."]
            else:
                add_ = [".0.", ".1.", ".2."]

            for abc in add_:

                if abc!=".":
                    state_dict['box_predictor' + abc + 'bbox_pred.weight'] = \
                        existing_model.roi_heads.box_predictor[int(abc[1:-1])].bbox_pred.weight
                    state_dict['box_predictor' + abc + 'bbox_pred.bias'] = \
                        existing_model.roi_heads.box_predictor[int(abc[1:-1])].bbox_pred.bias
                    state_dict['box_predictor' + abc + 'cls_score.weight'] = \
                        existing_model.roi_heads.box_predictor[int(abc[1:-1])].cls_score.weight[[2, 0, 80]]
                    state_dict['box_predictor' + abc + 'cls_score.bias'] = \
                        existing_model.roi_heads.box_predictor[int(abc[1:-1])].cls_score.bias[[2, 0, 80]]


                else:

                    state_dict['box_predictor.bbox_pred.weight'] = existing_model.roi_heads.box_predictor.bbox_pred.weight[[8,9,10,11, 0,1,2,3]]
                    state_dict['box_predictor.bbox_pred.bias'] = existing_model.roi_heads.box_predictor.bbox_pred.bias[[8,9,10,11, 0,1,2,3]]
                    state_dict['box_predictor' + abc + 'cls_score.weight'] = existing_model.roi_heads.box_predictor.cls_score.weight[[2, 0, 80]]
                    state_dict['box_predictor' + abc + 'cls_score.bias'] = existing_model.roi_heads.box_predictor.cls_score.bias[[2, 0, 80]]
            state_dict['mask_head.predictor.weight'] = existing_model.roi_heads.mask_head.predictor.weight[[2, 0]]
            state_dict['mask_head.predictor.bias'] = existing_model.roi_heads.mask_head.predictor.bias[[2, 0]]
            my_model.roi_heads.load_state_dict(state_dict)


            del existing_model
            del state_dict
        self.mrcnn_model=my_model.cuda(mrcnn_device0)
        self.mrcnn_model.roi_heads.cuda(mrcnn_device1)
        self.tracking_head=nn.Linear(8,1, bias=False).cuda(track_device)
        self.metadata=MetadataCatalog.get(my_cfg.DATASETS.TEST[0])

    def forward(self, images, labels, vid, path, image_names, cfg=None, Training=True, epoch=0, itr=0,location='.', use_mrcnn_loss=1, mrcnn_device0=0, mrcnn_device1=0):
        #mrcnn_predictions=[]
        #mrcnn_losses = []
        torch.cuda.empty_cache()
        Insts = []
        for lbs, image, img_name in zip(labels, images.squeeze(0), image_names):

            print(vid, img_name, len(lbs))
            if len(lbs):
                Inst = Instances(image.shape[1:])
                b = torch.stack([i["bbox"][0] for i in lbs])
                if b.dim() == 3:
                    b = b[:, 0, :]

                Inst.gt_boxes = Boxes(b)  # [Boxes(i["bbox"]) for i in lbs]
                Inst.gt_masks = BitMasks(
                    torch.stack([i["segmentation"][0] for i in lbs]))  # [i["segmentation"] for i in lbs]
                c = torch.stack([i["category_id"][0] for i in lbs])
                if c.dim() == 2:
                    c = c[:, 0]
                Inst.gt_classes = c  # .squeeze(1)
            else:
                Inst = None
            Insts.append(Inst)


        if Training:
            return self.forward_pass(images, Insts, mrcnn_device0, mrcnn_device1)
        else:
            #pdb.set_trace()
            with torch.no_grad():
                return self.forward_pass(images, Insts,mrcnn_device0, mrcnn_device1, training=False)


    def forward_pass(self,images, Insts,mrcnn_device0, mrcnn_device1, training=True):
        if training:
            batched_inputs = [{"image": img.float()[[2, 1, 0], :, :], "height": img.shape[1], "width": img.shape[2],
                               "instances": Inst}
                              if Inst != None else None for _, (img, Inst) in enumerate(zip(images.squeeze(0), Insts))]
            while None in batched_inputs:
                batched_inputs.remove(None)

            if batched_inputs != []:
                losses = self.mrcnn_losses(batched_inputs, mrcnn_device0=mrcnn_device0, mrcnn_device1=mrcnn_device1)
            else:
                losses = {'loss_cls': torch.tensor(0.).cuda(mrcnn_device1),
                          'loss_box_reg': torch.tensor(0.).cuda(mrcnn_device1),
                          'loss_mask': torch.tensor(0.).cuda(mrcnn_device1)}
        else:
            losses = {'loss_cls': torch.tensor(0.).cuda(mrcnn_device1),
                        'loss_box_reg': torch.tensor(0.).cuda(mrcnn_device1),
                        'loss_mask': torch.tensor(0.).cuda(mrcnn_device1)}

        batched_inputs = [{"image": img.float()[[2, 1, 0], :, :], "height": img.shape[1], "width": img.shape[2],
                           "instances": Inst} for _, (img, Inst) in enumerate(zip(images.squeeze(0), Insts))]
        if batched_inputs==[]:
            return [], [losses]
        mask_features, results = self.mrcnn_inference(batched_inputs, mrcnn_device0=mrcnn_device0,
                                                      mrcnn_device1=mrcnn_device1)

        pred = my_postprocessing(results, batched_inputs,
                                 [(img.shape[1], img.shape[2]) for img in images.squeeze(0)])
        ct = [len(p["instances"].pred_boxes) for p in pred]
        curr = 0
        for i in range(0, len(pred)):
            pred[i]["features"] = mask_features[curr:ct[i] + curr]
            curr += ct[i]

        #pdb.set_trace()

        return pred, [losses]


            #self.mrcnn_model.eval()
            #results = self.mrcnn_model.inference(batched_inputs, do_postprocess=False)


            #losses, mask_features_eval, results = self.mrcnn_custom_forward(batched_inputs, mrcnn_device0=mrcnn_device0, mrcnn_device1=mrcnn_device1)

            ############# mrcnn_custom_forward() starts

            # imgs = self.mrcnn_model.preprocess_image(batched_inputs)
            #
            # if "instances" in batched_inputs[0]:
            #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # else:
            #     gt_instances = None
            #
            # features = self.mrcnn_model.backbone(imgs.tensor)
            # proposals, proposal_losses = self.mrcnn_model.proposal_generator(imgs, features, gt_instances)
            #
            # for p in proposals:
            #     p.proposal_boxes.tensor=p.proposal_boxes.tensor.cuda(mrcnn_device1)
            #     p.objectness_logits=p.objectness_logits.cuda(mrcnn_device1)
            #
            # for f in features.keys():
            #     features[f]=features[f].cuda(mrcnn_device1)
            # pdb.set_trace()
            #
            # #_, detector_losses = self.roi_heads(imgs, features, proposals, gt_instances)
            # proposals = self.label_and_sample_proposals(proposals, gt_instances)
            #
            # instances = self.mrcnn_model.roi_heads._forward_box(features, proposals)
            # #instances = self.mrcnn_model.roi_heads._forward_mask(features, instances)
            # features = [features[f] for f in self.mrcnn_model.roi_heads.mask_in_features]
            # pred_boxes = [x.pred_boxes for x in instances]
            # mask_features = self.mrcnn_model.roi_heads.mask_pooler(features, pred_boxes)
            # results = self.mrcnn_model.roi_heads.mask_head(mask_features, instances)

            ############# mrcnn_custom_forward() ends





        # if Training:
        #     # self.mrcnn_model.eval()
        #     #pred = self.mrcnn_model([{"image":image.float()[[2,1,0],:,:],"height":image.shape[1],"width":image.shape[2]}])
        #         # for eval mode image is RGB
        #
        #     #if use_mrcnn_loss and len(pred[0]["instances"].pred_boxes) and Inst is not None:
        #     self.mrcnn_model.train()
        #     with EventStorage() as storage:
        #         batched_inputs = [{"image": img.float()[[2, 1, 0], :, :], "height": img.shape[1], "width": img.shape[2], "instances": Inst}
        #                           if Inst!=None else None for _, (img, Inst) in enumerate(zip(images.squeeze(0), Insts))]
        #         while None in batched_inputs:
        #             batched_inputs.remove(None)
        #
        #         if batched_inputs!=[]:
        #             pred_losses=self.mrcnn_model(batched_inputs)
        #         else:
        #             pred_losses = {'loss_cls': torch.tensor(0.).cuda(mrcnn_device1),
        #                            'loss_box_reg': torch.tensor(0.).cuda(mrcnn_device1),
        #                            'loss_mask': torch.tensor(0.).cuda(mrcnn_device1),
        #                            'loss_rpn_cls': torch.tensor(0.).cuda(mrcnn_device1),
        #                            'loss_rpn_loc': torch.tensor(0.).cuda(mrcnn_device1)}
        #
        # return pred, [losses]#, predicted_tracks, predicted_loss

    def mrcnn_losses(self, batched_inputs, mrcnn_device0=0, mrcnn_device1=0, cascade=False):
        self.mrcnn_model.train()

        imgs = self.mrcnn_model.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(mrcnn_device0) for x in batched_inputs]
        else:
            gt_instances = None
        features = self.mrcnn_model.backbone(imgs.tensor)
        with EventStorage() as storage:
            proposals, proposal_losses = self.mrcnn_model.proposal_generator(imgs, features, gt_instances)

            for p in proposals:
                p.proposal_boxes.tensor = p.proposal_boxes.tensor.cuda(mrcnn_device1)
                p.objectness_logits = p.objectness_logits.cuda(mrcnn_device1)

            for f in features.keys():
                features[f] = features[f].cuda(mrcnn_device1)

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(mrcnn_device1) for x in batched_inputs]
            else:
                gt_instances = None

           #pdb.set_trace()

            _, detector_losses = self.mrcnn_model.roi_heads(imgs, features, proposals, gt_instances)

            losses = {}
            losses.update(detector_losses)
            #losses.update(proposal_losses)
        return losses

    def mrcnn_inference(self, batched_inputs, mrcnn_device0=0, mrcnn_device1=0):
        self.mrcnn_model.eval()
        imgs = self.mrcnn_model.preprocess_image(batched_inputs)
        features = self.mrcnn_model.backbone(imgs.tensor)
        proposals, _ = self.mrcnn_model.proposal_generator(imgs, features, None)

        for p in proposals:
            p.proposal_boxes.tensor = p.proposal_boxes.tensor.cuda(mrcnn_device1)
            p.objectness_logits = p.objectness_logits.cuda(mrcnn_device1)

        for f in features.keys():
            features[f] = features[f].cuda(mrcnn_device1)

        instances = self.mrcnn_model.roi_heads._forward_box(features, proposals)
        instances = self.mrcnn_model.roi_heads._forward_mask(features, instances)
        features = [features[f] for f in self.mrcnn_model.roi_heads.mask_in_features]
        pred_boxes = [x.pred_boxes for x in instances]
        mask_features_eval = self.mrcnn_model.roi_heads.mask_pooler(features, pred_boxes)
        results = self.mrcnn_model.roi_heads.mask_head(mask_features_eval, instances)

        #pdb.set_trace()
        return mask_features_eval, results

    def  mrcnn_custom_forward(self, batched_inputs, mrcnn_device0=0, mrcnn_device1=1):
        imgs = self.mrcnn_model.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(mrcnn_device0) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.mrcnn_model.backbone(imgs.tensor)

        self.mrcnn_model.train()
        with EventStorage() as storage:
            proposals, proposal_losses = self.mrcnn_model.proposal_generator(imgs, features, gt_instances)

            for p in proposals:
                p.proposal_boxes.tensor = p.proposal_boxes.tensor.cuda(mrcnn_device1)
                p.objectness_logits = p.objectness_logits.cuda(mrcnn_device1)

            for f in features.keys():
                features[f] = features[f].cuda(mrcnn_device1)


            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(mrcnn_device1) for x in batched_inputs]
            else:
                gt_instances = None

            proposals1 = self.mrcnn_model.roi_heads.label_and_sample_proposals(proposals, gt_instances) # train
            losses, instances= self.my_forward_box(features, proposals1, proposals) #train (proposals1) #eval (proposals)

            features = [features[f] for f in self.mrcnn_model.roi_heads.mask_in_features]

            proposals1, _ = select_foreground_proposals(proposals1, self.mrcnn_model.roi_heads.num_classes) # train
            proposal_boxes = [x.proposal_boxes for x in proposals1] # train

            pred_boxes = [x.pred_boxes for x in instances]  # eval

            mask_features_train = self.mrcnn_model.roi_heads.mask_pooler(features, proposal_boxes) # train
            mask_features_eval = self.mrcnn_model.roi_heads.mask_pooler(features, pred_boxes)  # eval

            mask_loss, results = self.my_mask_head(mask_features_train, proposals1, mask_features_eval, instances)
            losses.update(mask_loss)  # train

        return losses, mask_features_eval, results

    def my_forward_box(self, features, proposals1, proposals): #proposals1 for losses, proposals for op
        features = [features[f] for f in self.mrcnn_model.roi_heads.box_in_features]
        box_features = self.mrcnn_model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.mrcnn_model.roi_heads.box_head(box_features)
        predictions = self.mrcnn_model.roi_heads.box_predictor(box_features)

        box_features1 = self.mrcnn_model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals1])
        box_features1 = self.mrcnn_model.roi_heads.box_head(box_features1)
        predictions1 = self.mrcnn_model.roi_heads.box_predictor(box_features1)

        losses = self.mrcnn_model.roi_heads.box_predictor.losses(predictions1, proposals1)

        pred_instances, _ = self.mrcnn_model.roi_heads.box_predictor.inference(predictions, proposals)

        return losses, pred_instances

    def my_mask_head(self, mask_features_train, proposals1, mask_features_eval, instances):
        x1 = self.mrcnn_model.roi_heads.mask_head.layers(mask_features_train)
        loss= {"loss_mask": mask_rcnn_loss(x1, proposals1)}
        x = self.mrcnn_model.roi_heads.mask_head.layers(mask_features_eval)
        inst=mask_rcnn_inference(x, instances)

        return loss, inst





def my_postprocessing(instances, batched_inputs, image_sizes):
    processed_results = []
    for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
    ):
        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])
        r = my_detector_postprocess(results_per_image, height, width)
        processed_results.append({"instances": r})
    return processed_results


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.
    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()

def my_paste_masks_in_image(masks, boxes, image_shape, threshold=0.5):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.
    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.
    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.
    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """

    BYTES_PER_FLOAT = 4
    # TODO: This memory limit may be too much or too little. It would be better to
    # determine it based on available resources.
    GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit

    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu":
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
                num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.float if threshold >= 0 else torch.uint8
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )

        # if threshold >= 0:
        #     masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        # else:
        #     # for visualization and debugging
        #     masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks


def my_detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.
    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.
    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        results.pred_masks = my_paste_masks_in_image(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results

# args = default_argument_parser().parse_args()
# cfg=setup(args)
# model=MOTS_model(cfg)
# print("hi")

# class REID_model(nn.Module):
#     def __init__(self, device=0):
#         super(REID_model, self).__init__()
#         self.conv1=nn.Conv2d(256,128, 3, bias=False)
#         self.bn1=nn.BatchNorm2d(128)
#         self.conv2=nn.Conv2d(128,64, 3, bias=False)
#         self.bn2=nn.BatchNorm2d(64)
#         self.conv3=nn.Conv2d(64,16, 3, bias=False)
#         self.bn3=nn.BatchNorm2d(16)
#
#         self.layer1=nn.Linear(8*8*16,512, bias=False)
#
#         self.relu=nn.ReLU()
#
#         #self.layer2 = nn.Linear(5000, 500, bias=False).cuda(device)
#     def reid_net(self,r, device=0):
#         r=r.reshape(r.shape[0],256,14,14).cuda(device)
#         r=self.relu(self.bn1(self.conv1(r)))
#         r=self.relu(self.bn2(self.conv2(r)))
#         r=self.relu(self.bn3(self.conv3(r)))
#         r=self.layer1(r.reshape(r.shape[0],r.shape[1]*r.shape[2]*r.shape[3]))
#         return r
#
#     def forward(self, reids, same_reids, diff_reids, device=0):
#         r=self.reid_net(reids, device=device)
#         same_r=self.reid_net(same_reids,device=device)
#         diff_r = self.reid_net(diff_reids,device=device)
#
#         triplet_loss =nn.TripletMarginWithDistanceLoss(distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y))
#
#         #triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
#
#         loss = triplet_loss(r, same_r, diff_r)
#
#         return loss

class REID_model(nn.Module):
    def __init__(self, car=False):
        super(REID_model, self).__init__()

        reid_model = torch.nn.DataParallel(TrackerOffsetEmb(num_points=1000, margin=0.2,
                                                            env_points=500, border_ic=3, outputD=32, category=True))
        if car==True:
            state = torch.load('../../PointTrack_undisturbed/car_finetune_tracking/checkpoint.pth')
        else:
            state = torch.load('../../PointTrack_undisturbed/person_finetune_tracking/checkpoint.pth')
        reid_model.load_state_dict(state['model_state_dict'], strict=True)
        self.reid_model = reid_model
        del reid_model, state
        if car==True:
            self.dataset = MOTSTrackCarsValOffset(type="val", num_points=1500, gt=False, box=True, category=True)
        else:
            self.dataset=MOTSTrackPersonValOffset(type="val",num_points=1500,gt=False, box=True,  category=True)



    def forward(self, images, op_mask, seg_device1=0):
        #seg_predictions=[]
        #seg_losses = []

        reid_samples=[]
        for _,(img, masks) in enumerate(zip(images[0].cpu().numpy(), op_mask)):
            instance_map = np.zeros((img.shape[-2],img.shape[-1])).astype(np.uint8)
            if masks!=[]:
                ct=1
                for mask in masks:
                    instance_map=instance_map+((mask>0.5).cpu().numpy().astype(np.uint8))*ct
                    ct+=1

            # instance_map = cluster.cluster_mots_wo_points(output, threshold=0.94,
            #                                               min_pixel=160,
            #                                               with_uv=True, n_sigma=2)
                sample=self.dataset.get_data_for_structured_mots(img.transpose(1,2,0)[:,:,[2,1,0]], instance_map)
                with torch.no_grad():
                    rd=self.get_reid(sample, device=seg_device1)
                reid_samples.append([r for r in rd])
            #(output[4] + (1 - output[4].detach())) * torch.tensor(sample["masks"]).cuda(seg_device1)
            else:
                reid_samples.append([])


        return reid_samples

    def get_reid(self,sample, device=0):
        points = sample['points']
        if len(points) < 1:
            reids = np.array([])
        else:
            xyxys = sample['xyxys']
            reids = self.reid_model(torch.tensor([points]), None, torch.tensor([xyxys]).cuda(device), infer=True)

        return reids
