
from .utils import mots_helper, file_helper
from .datasets import dataset
import scipy.optimize as opt

import numpy as np
import pickle
import os
import pdb
import time
import os.path
import torch
import cv2
import networkx as nx
from copy import deepcopy

class DetectionSpace():
    def __init__(self, args, seq, reid_model, train=False):
        self.keep_alive=45
        self.args=args
        self.seq=seq
        self.reid_model=reid_model

    def get_images(self, images):
        if self.args.mots:
            images = torch.stack([img.squeeze(0) for img in images]).unsqueeze(dim=0)
            images = dataset.pad_and_stack_images(images).cuda(self.args.seg_device0)

            self.images_shape=images.shape
            self.images=images[0].cuda(self.args.track_device)

        else:
            self.images=images
            np_img = cv2.imread(images[0][0])
            self.images_shape=(1, len(images), np_img.shape[2], np_img.shape[0], np_img.shape[1])
        return

    def get_optical_flow(self, image_names):
        self.n_optflow_skip0=[]
        self.n_optflow_skip1=[]
        self.image_names=image_names

        if self.args.use_optical_flow is True:
            self.n_optflow_skip0, self.n_optflow_skip1 = mots_helper.get_optical_flow_locations(self.args, self.seq, self.image_names)

        else:
            self.n_optflow_skip0=[[] for i in range(0,len(self.image_names)-1)]
            self.n_optflow_skip1 = [[] for i in range(0, len(self.image_names) - 2)]

        return

    def get_detections(self, image_names=[]):  # clean up

        self.videos, start_at_1=file_helper.dataset_specifics_test(self.args)

        if self.args.det_dir.find("trackrcnn")!= -1:
            self.bb_format = "trackrcnn" # bounding box format
        else:
            self.bb_format = "others" # bounding box format

        if self.args.car:
            self.classes_to_load = [1]
        else:
            self.classes_to_load = [2]

        if self.args.use_given_detections and self.reid_model == []:

            if self.args.mots:
                self.boxes, self.scores, self.reids, self.classes, self.masks, self.ts = file_helper.import_detections_for_sequence(self, names=image_names)
            else:
                self.boxes, self.scores, self.reids, self.classes, _, self.ts = file_helper.import_detections_for_mot(self.videos[self.args.vid], os.path.join(self.args.det_dir, self.args.dataset),
                                                                                                 device=self.args.track_device, names=image_names,
                                                                                                 bb_format=self.bb_format, thresh=self.args.det_thresh)
                print("Making masks...")
                self.masks = mots_helper.make_mask_from_box_discriminative(self.boxes, self.images, self.images.shape[-2:], mots=self.args.mots)


        elif self.args.use_given_detections and self.reid_model != []:
            if self.args.mots:
                self.boxes, self.scores, _, self.classes, self.masks, self.ts = file_helper.import_detections_for_sequence(self, names=image_names,)
                self.masks = mots_helper.remove_overlap(self.masks, self.scores, train=False)
                print("Making reids...")
                self.reids = self.reid_model(self.images.unsqueeze(dim=0), self.masks, mots=self.args.mots)
            else:
                boxes, scores, _, classes, masks, ts = file_helper.import_detections_for_mot(self.videos[self.args.vid],os.path.join(
                                                                                                     self.args.det_dir,
                                                                                                     self.args.dataset),
                                                                                                 device=self.args.track_device,
                                                                                                 bb_format=self.bb_format,
                                                                                                thresh=self.args.det_thresh)
                masks = mots_helper.make_mask_from_box_discriminative(boxes, images, images_shape[-2:], mots=self.args.mots)
                reids = self.reid_model(self.images.unsqueeze(dim=0), masks, mots=self.args.mots)

                if self.train is False:
                    file = open("dets_" + vid[0] + ".pkl", 'wb')
                    pickle.dump([boxes, scores, classes, masks, ts],file)
                    file.close()
                    print("Making reids...")
                    #reids = self.reid_model(images, boxes)


        elif not self.args.use_given_detections:
            images=images.cpu()
            if self.reid_model==[] and self.args.seg_model_to_use=="MaskRCNN50":
                raise RuntimeError("Please specify re-identification model path")

            print("Seg Predictions...")
            tseg = time.time()
            chunk_size = 10
            image_chunks = [images[0][(i) * chunk_size:(i + 1) * chunk_size].unsqueeze(0) for i in
                            range(0, int(len(images[0]) / chunk_size) + 1)]
            gt_label_chunks = [gt_labels[(i) * chunk_size:(i + 1) * chunk_size] for i in
                               range(0, int(len(images[0]) / chunk_size) + 1)]
            img_name_chunks = [image_names[(i) * chunk_size:(i + 1) * chunk_size] for i in
                               range(0, int(len(images[0]) / chunk_size) + 1)]

            seg_losses = 0.
            boxes=[]
            masks=[]
            scores=[]
            reids=[]
            classes=[]

            for i_new, (imgs, gt_lbs, img_nms) in enumerate(zip(image_chunks, gt_label_chunks, img_name_chunks)):

                if imgs.shape[1]>0:
                    boxes_, scores_, masks_, reids_, classes_, seg_losses_ = model(imgs.cuda(self.args.seg_device0), gt_lbs, car=self.args.car, Training=False)

                    boxes=boxes+boxes_
                    scores=scores+scores_
                    masks=masks+masks_
                    reids=reids+reids_
                    classes=classes+[torch.stack([c - 1 for c in ct]) if ct.shape[0]>1 else torch.tensor([c - 1 for c in ct]) for ct in classes_]
                    seg_losses = seg_losses + seg_losses_.cpu()
                    del boxes_,scores_, masks_, reids_, classes_, seg_losses_
                    torch.cuda.empty_cache()
                    print(i_new)


            print("Time: ", (time.time() - tseg))


            if self.args.save_files_only == True:
                path = "../Structured_MOTS/output/KITTI_MOTS/val/detections/detectron2_x152"
                out_path = '../pointTrack_segs_detectron2_R101_car_ep0/'
                class_save = 0

                if not os.path.exists(out_path):
                    os.mkdir(out_path)

                for i, (masks_t, classes_t, scores_t) in enumerate(zip(masks, classes, scores)):

                    if len(masks_t) > 0:

                        filename = out_path + vid[0] + "_" + str(i) + ".pkl"
                        ctr = 1
                        instance_map = np.zeros_like(masks_t[0].cpu().numpy()).astype(np.uint8)
                        k = 0
                        for i2, (mask, cls, sc) in enumerate(zip(masks_t, classes_t, scores_t)):

                            if cls.item() == class_save:
                                print(sc, cls)
                                k = 1
                                instance_map = instance_map + ((mask > 0.5).cpu().numpy().astype(np.uint8) * ctr)
                                ctr += 1

                        if k == 1:
                            with open(filename, 'wb') as f:
                                pickle.dump(instance_map, f, protocol=2)
                return

                # [pred["features"].reshape(pred["features"].shape[0],
                # pred["features"].shape[1]*pred["features"].shape[2]*
                # pred["features"].shape[3]).cuda(self.args.track_device) for pred in predictions]

        #masks = mots_helper.remove_overlap(masks,scores, train=False) #train=false given for reformatting back into lists

        self.boxes, self.scores, self.reids, self.classes, self.masks = mots_helper.remove_detections(self.boxes, self.scores, self.classes, self.masks, self.reids, mots=self.args.mots) # remove detections with very small area
        return

    def create_assignment_labels(self, labels, predictions_mots,det_class_ids=[2,0], mots=True):

        if mots:
            tracking_labels, predictions_mots_new = rearrange_gt(labels, predictions_mots,det_class_ids=det_class_ids)
        else: # mot
            tracking_labels, predictions_mots_new = rearrange_gt_mot(labels, predictions_mots)

        return tracking_labels, predictions_mots_new


    def get_tracks(self, assignments): # assign ids
        start = 0
        all_tracks = []
        for t, det_t in enumerate(self.boxes):
            tracks = []

            if start == 0:
                track_ids = np.array(range(1, len(det_t) + 1)).astype(np.int64)
                leaf_nodes = []
                track_ids_prev = np.array([])
                track_counter = len(det_t) + 1
                start = 1
            elif t > 0 and t < len(self.boxes):
                assignment= np.stack(np.where(assignments.G.nodes[str(t - 1) + "_" + str(assignments.y_star[t-1])]["node_assignment"].cpu()), axis=1)
                track_ids_prev2 = track_ids_prev
                track_ids_prev = track_ids
                track_ids = np.zeros((len(det_t)), dtype=np.int64)
                new_id_list = []
                for id in range(0, len(det_t)):
                    new_id = 1
                    if id in assignment[:, 1] and len(track_ids_prev) > 0:
                        pos = np.where(assignment[:, 1] == id)
                        if self.args.train is True:
                            condition=True
                        else:
                            condition=assignments.G.nodes[str(t - 1) + "_" + str(assignments.y_star[t - 1])]["cost_matrix_iou"][assignment[pos, 0][0][0]][
                                assignment[pos, 1][0][0]] < 1
                        if condition:
                            track_ids[id] = int(track_ids_prev[assignment[pos, 0]])
                            new_id = 0

                    elif t > 1 and self.args.K> 1 and self.args.second_order:
                        if len(self.boxes[t - 2]) > len(self.boxes[t - 1]) and len(self.boxes[t - 1]) < len(det_t):
                            cost_matrix1_iou=assignments.G.nodes[str(t - 1) + "_" + str(assignments.y_star[t - 1])]["cost_matrix_iou_2nd_order"]
                            find_extras = np.where(cost_matrix1_iou[:,id] < 1)[0]
                            if len(find_extras) > 0 and id in find_extras:
                                arg_extra = find_extras[
                                    np.argmin(cost_matrix1_iou[np.where(cost_matrix1_iou[:,id] < 1)[0], id])]
                                if track_ids_prev2[arg_extra] not in track_ids and track_ids_prev2[
                                    arg_extra] not in track_ids_prev:
                                    track_ids[id] = track_ids_prev2[arg_extra]

                                    leaf_nodes.append({"name":str(t) + "_" + str(id), "box": self.boxes[t][id], "reid": self.reids[t][id]})
                                    if str(t - 2) + "_" + str(arg_extra) in [l_n["name"] for l_n in leaf_nodes]:
                                        leaf_nodes.remove({"name":str(t - 2) + "_" + str(arg_extra), "box": self.boxes[t-2][arg_extra], "reid": self.reids[t-2][arg_extra]})
                                        # leaf_nodes.remove(track_ids_prev2[arg_extra]) #remove from leaf nodes because edge is drawn from it now

                                    new_id = 0

                    if new_id == 1:
                        new_id_list.append(id)
                        new_id = 0


                for id in new_id_list:
                    # print(t, id)

                    new_name = -1


                    if new_name == -1:
                        new_name = int(track_counter)
                        track_counter += 1
                        track_ids[id] = int(new_name)


            all_tracks.append(list(track_ids))

        tracklets=mots_helper.get_tracklets(all_tracks, self.reids, self.boxes)

        reids = [torch.stack([tr["reids"] for tr in tracks]) for tracks in tracklets]
        boxes = [torch.stack([tr["box"] for tr in tracks]) for tracks in tracklets]
        #avg_reids = torch.stack([torch.sum(rd, dim=0) for rd in reids])
        avg_reids_begin=torch.stack([torch.mean(rd[:5], dim=0) for rd in reids])
        avg_reids_end = torch.stack([torch.mean(rd[-5:], dim=0) for rd in reids])
        ts = [[tr["t"] for tr in tracks] for tracks in tracklets]
        track_ids_all = [[tr["track_id"] for tr in tracks] for tracks in tracklets]

        #dist_matrix = torch.cdist(avg_reids.unsqueeze(0),avg_reids.unsqueeze(1)).squeeze(2)# np.zeros((len(tracklets), len(tracklets)))
        reid_matrix = 1-torch.mm(avg_reids_end, avg_reids_begin.permute(1, 0))

        #dist_matrix = torch.cdist(avg_reids_end.unsqueeze(0),avg_reids_begin.unsqueeze(1)).squeeze(2)
        #cost_matrix = munkres.make_cost_matrix(dist_matrix)
        for i in range(len(reid_matrix)):
            #these are the beginnings

            reid_matrix[i,i]=100000 # preventing these values
            reid_matrix[i,np.where(ts[i][0]<=np.array([t[-1] for t in ts]))]=100000 # preventing these values
            reid_matrix[i, np.where(ts[i][0] > (np.array([t[-1] for t in ts])+self.keep_alive))] = 100000  # preventing these values

        #cost_matrix
        indexes_temp = opt.linear_sum_assignment(reid_matrix.cpu())
        indexes=[]
        for pairs in np.array(indexes_temp).transpose():
            if reid_matrix[pairs[0], pairs[1]]<0.9:
                if not self.args.mots:
                    b1 = boxes[pairs[1]][-1]
                    b2 = boxes[pairs[0]][0]
                    condition1=abs(b1[2]/b1[3] -b2[2]/b2[3]) <0.2 #aspect_ratio

                    box_diff=b2-b1
                    tdiff = ts[pairs[0]][0] - ts[pairs[1]][-1]
                    condition2=((box_diff[0]/tdiff < 40) and (box_diff[1]/tdiff < 40) and (box_diff[2]/tdiff < 40) and (box_diff[3]/tdiff < 40))
                    condition= condition1 and condition2
                else:
                    condition =True
                if condition:
                    indexes.append(pairs)

        groups=mots_helper.get_groups(np.array(indexes))
        all_tracks_new=all_tracks.copy()
        for g in groups:
            dominant_track=tracklets[g[0]][0]["track_id"]
            all_tracks_new=[[tr if tr not in [tri["track_id"] for i in g[1:] for tri in tracklets[i]]
                             else dominant_track for tr in tracks_t] for tracks_t in all_tracks_new]

        self.track_ids = all_tracks_new

        return


    def get_tracks_for_next_frame(self, dets, t, interval):
        if t==0:
            self.track_ids=[dets.track_ids[0]]

        print("t: ", t, dets.track_ids)


        #new_track_ids=  [self.track_ids[-1][m.item()] if m.tolist()!=[] else self.max_id+1 for m in [torch.where(torch.tensor(dets.track_ids[0])==k)[0] for k in dets.track_ids[1]]]

        self.track_ids.append(self.get_new_track_ids(dets))
        print(self.track_ids)

        if len(self.track_ids)>len(self.boxes)-interval+1:

            for t1 in range(1,interval-1):
                print("t: ", t+t1)
                self.track_ids.append(self.get_new_track_ids(dets))

        return


    def get_new_track_ids(self, dets):
        new_track_ids= []
        self.max_id=max([a for b in self.track_ids for a in b])

        for k in dets.track_ids[1]:
            m=torch.where(torch.tensor(dets.track_ids[0])==k)[0]

            if m.tolist()!=[]:
                if dets.t1==0:
                    new_track_ids.append(dets.track_ids[0][m.item()])
                else:
                    new_track_ids.append(dets.original_track_ids[0][m.item()])
            else:
                new_track_ids.append(self.max_id+1)
                self.max_id=self.max_id+1

        return new_track_ids

class miniDetectionSpace(DetectionSpace):
    def __init__(self, detections, t, frame_gap):
        super().__init__(detections.args, detections.seq, detections.reid_model, train=detections.args.train)

        self.t1=t
        self.t2=t+frame_gap
        self.detections=detections
        self.bb_format=detections.bb_format
        self.classes_to_load=detections.classes_to_load
        self.images_shape=detections.images_shape

    def extract_few_frames(self, carry_forward=[]):
        self.image_names=self.detections.image_names[self.t1:self.t2]
        self.images=self.detections.images[self.t1:self.t2]
        self.n_optflow_skip0=self.detections.n_optflow_skip0[self.t1:self.t2]
        self.n_optflow_skip1=self.detections.n_optflow_skip1[self.t1:self.t2]
        self.boxes=self.detections.boxes[self.t1:self.t2]
        self.masks=self.detections.masks[self.t1:self.t2]
        self.scores=self.detections.scores[self.t1:self.t2]
        self.reids=self.detections.reids[self.t1:self.t2]
        self.classes=self.detections.classes[self.t1:self.t2]
        self.ts=self.detections.ts[self.t1:self.t2]
        if hasattr(self.detections, "track_ids") is True:
            self.original_track_ids=self.detections.track_ids[self.t1:]

        if carry_forward!=[]:
            self.boxes[0]=self.boxes[0] + [k["boxes"] for k in carry_forward]
            self.classes[0]=self.classes[0]+[k["classes"] for k in carry_forward]
            self.reids[0]=self.reids[0]+[k["reids"] for k in carry_forward]
            self.ts[0]=self.ts[0]+[k["ts"] for k in carry_forward]
            self.scores[0]=self.scores[0]+[k["scores"] for k in carry_forward]
            self.masks[0]=self.masks[0]+ [k["masks"] for k in carry_forward]
            self.original_track_ids[0]=self.original_track_ids[0]+[k["original_track_ids"] for k in carry_forward]


        return
