
import torch
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pycocotools.mask as cocomask
import pdb
import time
import cv2
import torch.utils.data as data
import os
import os.path
import sys
from external.mots_tools.mots_common import io as io1


def get_mots_dict(data_dir, vid="all"):

    if vid == "all":
        all_dirs = sorted(os.listdir(os.path.join(data_dir, "images")))
    else:
        all_dirs = sorted(os.listdir(os.path.join(data_dir, "images")))[vid:vid + 1]


    idx=0
    dataset_dicts = []
    for dir1 in all_dirs:
        image_names = sorted(os.listdir(os.path.join(data_dir, "images", dir1)))
        if os.path.exists(os.path.join(data_dir, "instances_txt")):
            objects_per_frame = io1.load_txt(os.path.join(data_dir, "instances_txt", dir1 + ".txt"))
        else:
            objects_per_frame=[]
        for i in range(0,len(image_names)):
            record={}

            image_name = image_names[i]
            image_path = os.path.join(os.path.join(data_dir, "images", dir1, image_name))
            np_img = cv2.imread(image_path)

            record["file_name"] = image_path
            record["image_name"] = image_name
            record["video"] = dir1
            record["image_id"] = idx
            record["height"] = np_img.shape[0]
            record["width"] = np_img.shape[1]
            del np_img

            if objects_per_frame!=[] and int(image_name[:-4]) in list(objects_per_frame.keys()):
                labels=[]
                for obj in objects_per_frame[int(image_name[:-4])]:

                    labels.append({"bbox": torch.tensor([cocomask.toBbox(obj.mask)[0],
                                       cocomask.toBbox(obj.mask)[1],
                                       cocomask.toBbox(obj.mask)[0] + cocomask.toBbox(obj.mask)[2],
                                       cocomask.toBbox(obj.mask)[1] + cocomask.toBbox(obj.mask)[3]]),
                     # cocomask.toBbox(obj.mask),
                     "segmentation": cocomask.decode(obj.mask),
                     "category_id": 0 if obj.class_id == 1 else 1 if obj.class_id == 2 else 2, #car, ped, ignore
                     "ignore":2, # ignore class category will be 2

                     # converting to format: 0 for car, 1 for ped
                     "track_id": obj.track_id})
            else:
                labels=[]

            record["annotations"] = labels
            dataset_dicts.append(record)
            idx+=1

    return dataset_dicts


class MOTS_dataset(Dataset):
    def __init__(self, mots_dict, n, train=True, transform=None):
        self.transform = transform
        if train==True:
            self.n = n
        self.LoadData(mots_dict, train=train)
        assert len(self.labels) == len(self.images)

    def LoadData(self, mots_dict, train=True):
        if train:
            n = self.n
        self.images = []
        self.labels = []
        self.vid = []
        self.image_names = []
        videos=list(set([i["video"] for i in mots_dict]))

        for vid in videos:
            dicts_for_current_video=[i for i in mots_dict if i["video"]==vid]
            if train==False:
                n=len(dicts_for_current_video)

            n_images=[]
            n_labels=[]
            n_img_names=[]
            i=0
            while i < len(dicts_for_current_video):
                counter=0
                while counter<n and i < len(dicts_for_current_video):

                    image_name=dicts_for_current_video[i]["image_name"]
                    image_path=dicts_for_current_video[i]["file_name"]
                    np_img=cv2.imread(image_path)
                    img_temp=torch.from_numpy(np_img).permute(2,0,1)

                    n_images.append(img_temp[[2,1,0],:,:])
                    n_img_names.append(image_name)
                    temp_anno=dicts_for_current_video[i]["annotations"].copy()

                    n_labels.append(temp_anno)

                    counter+=1
                    i += 1

                if len(n_images)==n:
                    self.images.append(n_images)
                    self.labels.append(n_labels)
                    self.vid.append(vid)
                    print(vid, n_img_names[-1])
                    self.image_names.append(n_img_names)
                n_images = []
                n_labels = []
                n_img_names = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return ( self.images[idx],self.labels[idx], self.vid[idx], self.image_names[idx])


def pad_and_stack_images(images):

    h=np.array([i.shape[2] for i in images]).max()
    w=np.array([i.shape[3] for i in images]).max()
    images_new=[]

    for img in images:
        if img.shape[2]<h:
            pad=torch.zeros(img.shape[0],img.shape[1],h-img.shape[2],img.shape[3], dtype=torch.uint8)
            img=torch.cat((img, pad), 2)

        if img.shape[3]<w:
            pad = torch.zeros(img.shape[0], img.shape[1], img.shape[2], w-img.shape[3], dtype=torch.uint8)
            img = torch.cat((img, pad), 3)

        images_new.append(img)

    return torch.stack(images_new)
