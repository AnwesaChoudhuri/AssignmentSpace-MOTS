from collections import OrderedDict

import torch
import torch.nn.functional as F

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection import maskrcnn_resnet50_fpn
import pycocotools.mask as cocomask
import numpy as np
import pdb
class MRCNN_FPN():

    def __init__(self):
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.original_image_sizes = None
        self.model.preprocessed_images = None
        self.model.features = None

    def predict_masks(self, boxes, images):

        device = list(self.model.parameters())[0].device
        boxes = boxes.to(device)
        images = images.to(device)

        self.model.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.model.transform(images, None)
        self.model.preprocessed_images = preprocessed_images

        self.model.features = self.model.backbone(preprocessed_images.tensors)
        if isinstance(self.model.features, torch.Tensor):
            self.model.features = OrderedDict([(0, self.model.features)])

        boxes_new = resize_boxes(boxes, self.model.original_image_sizes[0], self.model.preprocessed_images.image_sizes[0])
        proposals = [boxes_new]

        mask_features = self.model.roi_heads.mask_roi_pool(self.model.features, proposals, self.model.preprocessed_images.image_sizes)

        pred_mask_heads = self.model.roi_heads.mask_head(mask_features)
        pred_masks = self.model.roi_heads.mask_predictor(pred_mask_heads)

        preds=[{"boxes": boxes_new, "masks": pred_masks[:,1:2], "labels": torch.tensor([1 for k in pred_masks]),"scores": torch.tensor([1. for k in pred_masks])}]
        preds_new= self.model.transform.postprocess(preds, self.model.preprocessed_images.image_sizes, self.model.original_image_sizes)
        pred_masks=preds_new[0]["masks"].squeeze(1).permute(1,2,0).detach().cpu().numpy()>0.5
        return cocomask.encode(np.asfortranarray(pred_masks.astype(np.uint8)))
