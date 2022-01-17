import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image

from torch.autograd import Variable
import os
import pdb
import pycocotools.mask as cocomask
from ..datasets import dataset as structured_dataset

from external.PointTrack.models.BranchedERFNet import TrackerOffsetEmb
from external.PointTrack.datasets.KittiMOTSDataset import *

from external.DGNetPP.reIDmodel import ft_net, ft_netAB, ft_net_dense, PCB, PCB_test


class REID_model(nn.Module):
    def __init__(self, path=".",car=False):
        super(REID_model, self).__init__()

        reid_model = torch.nn.DataParallel(TrackerOffsetEmb(num_points=1000, margin=0.2,
                                                            env_points=500, border_ic=3, outputD=32, category=True))
        state = torch.load(path)
        reid_model.load_state_dict(state['model_state_dict'], strict=False) #Be careful here! strict=True might not work because batch norm layers aren't loaded here
        self.reid_model = reid_model
        del reid_model, state
        if car==True:
            self.dataset = MOTSTrackCarsValOffset(type="val", num_points=1500, gt=False, box=True, category=True)
        else:
            self.dataset=MOTSTrackPersonValOffset(type="val",num_points=1500,gt=False, box=True,  category=True)

    def forward(self, images, op_mask, seg_device1=0, mots=True):

        #seg_predictions=[]
        #seg_losses = []

        reid_samples=[]
        if mots:
            all_images=images[0].cpu().numpy()
        else:
            all_images=images.copy()
        for _,(img, masks) in enumerate(zip(all_images, op_mask)):
            if not mots:
                np_img = cv2.imread(img[0])
                img_temp = torch.from_numpy(np_img).permute(2, 0, 1)[[2, 1, 0], :, :]

                img = img_temp.unsqueeze(dim=0).unsqueeze(0)
                img = structured_dataset.pad_and_stack_images(img).cuda(seg_device1)[0][0].cpu().numpy()

            instance_map = np.zeros((img.shape[-2],img.shape[-1])).astype(np.uint8)

            if masks!=[]:
                if not mots:
                    masks=torch.tensor(cocomask.decode(masks)).permute(2,0,1)
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


class REID_model_mot(nn.Module):
    def __init__(self, path="."):

        super(REID_model_mot, self).__init__()

        model_path, output_dim = self.get_model_stats()

        model = ft_netAB(output_dim, norm=False, stride=1, pool=max)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['a'], strict=False)
        model.model.fc = nn.Sequential()
        model.classifier1.classifier = nn.Sequential()
        model.classifier2.classifier = nn.Sequential()

        self.reid_model = model

        self.transforms=transforms.Compose([
            transforms.Resize((256,128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        del model

    def forward(self, imgs, boxes, PCB=False):
        self.reid_model.eval()
        all_features=[]

        for i_, (image, boxes_t) in enumerate(zip(imgs, boxes)):
            image=cv2.imread(image[0])
            features = torch.FloatTensor()
            count = 0
            imgs=[]
            for box in boxes_t:
                img=image[int(box[1]):int(box[3]), int(box[0]):int(box[2]),:]
                img=self.transforms(Image.fromarray(img))
                imgs.append(img)

            ff = torch.FloatTensor(0, 1024).zero_()
            if PCB:
                ff = torch.FloatTensor(0, 2048, 6).zero_()  # we have six parts
            if imgs!=[]:
                imgs=torch.stack(imgs)
                n, c, h, w = imgs.size()
                count += n
                # print(count)
                ff = torch.FloatTensor(n, 1024).zero_()
                if PCB:
                    ff = torch.FloatTensor(n, 2048, 6).zero_()  # we have six parts
                for i in range(2):
                    if (i == 1):
                        imgs = self.fliplr(imgs)
                    input_imgs = Variable(imgs.cuda())

                    with torch.no_grad():
                        f, x = self.reid_model(input_imgs)

                    x[0] = self.norm(x[0])
                    x[1] = self.norm(x[1])
                    f = torch.cat((x[0], x[1]), dim=1)  # use 512-dim feature
                    f = f.data.cpu()
                    ff = ff + f

                ff[:, 0:512] = self.norm(ff[:, 0:512])
                ff[:, 512:1024] = self.norm(ff[:, 512:1024])

                # norm feature
                if PCB:
                    # feature size (n,2048,6)
                    # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
                    # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
                    ff = ff.div(fnorm.expand_as(ff))
                    ff = ff.view(ff.size(0), -1)

            features = torch.cat((features, ff), 0)
            all_features.append(features.cuda(0))
        return all_features

    def norm(self,f):
        if f.shape[0]>1:
            f = f.squeeze()
        fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
        f = f.div(fnorm.expand_as(f))
        return f

    def fliplr(self,img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def get_model_stats(self):
        checkpoint_name = 'id_00172353.pt'
        main_folder = os.path.join('DG-Net-PP/outputs', 'best-duke2market')
        folders = os.listdir(main_folder)
        for folder in folders:
            if not folder.isdigit():
                continue
            checkpoint_folder = os.path.join(main_folder, folder, 'checkpoints')
            checkpoints = os.listdir(checkpoint_folder)
            if checkpoint_name in checkpoints:
                checkpoint_path = os.path.join(checkpoint_folder, checkpoint_name)
                checkpoint = torch.load(checkpoint_path)
                return checkpoint_path, checkpoint['a']['classifier2.classifier.0.weight'].size()[0]
        del checkpoint
        torch.cuda.empty_cache()

        print('No checkpoint found.')


class REID_model_mot_resnet(nn.Module):
    def __init__(self):

        super(REID_model_mot_resnet, self).__init__()
        sys.path.insert(0, os.path.abspath('tracking_wo_bnw/src'))
        from tracktor.reid.resnet import ReIDNetwork_resnet50
        pdb.set_trace()
        reid_state = torch.load("/home/a-m/anwesac2/MOTS/mot_neural_solver/resnet50_market_cuhk_duke.tar-232",map_location=lambda storage, loc: storage)
        #reid_state = torch.load("/home/a-m/anwesac2/Tracking/tracking_wo_bnw/output/tracktor/reid/res50-mot17-batch_hard/ResNet_iter_25245.pth",map_location=lambda storage, loc: storage)
        reid_model = ReIDNetwork_resnet50(output_dim=128, loss="batch_hard", margin=0.2, prec_at_k=3, crop_H=256, crop_W=128,normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225], pretrained=False)
        self.reid_model = reid_model

        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in reid_state["state_dict"].items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # model.load_state_dict(new_state_dict)

        self.reid_model.load_state_dict(reid_state)

        self.reid_model.eval()
        self.reid_model=self.reid_model.cuda()

    def forward(self, imgs, boxes):
        all_features = []
        for i_, (image, boxes_t) in enumerate(zip(imgs, boxes)):
            features= torch.FloatTensor(0, 128).zero_()
            if len(boxes_t)>0:
                image = cv2.imread(image[0])
                image = torch.tensor(image).permute(2,0,1).unsqueeze(0)
                with torch.no_grad():
                    features = self.reid_model.test_rois(image, boxes_t)
            all_features.append(features.cuda(0))

        return all_features
