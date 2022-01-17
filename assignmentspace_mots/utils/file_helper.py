import numpy as np
from collections import namedtuple
import pycocotools.mask as cocomask
import os
import png
import torch
import cv2
from .mots_helper import get_optical_flow

import pdb


def export_tracking_result_in_kitti_format(tag, tracks, add_masks, model_str, out_folder="", start_time_at_1=False):
    if out_folder == "":
        out_folder = "forwarded/" + model_str + "/tracking_data"
    os.makedirs(out_folder, exist_ok=True)
    out_filename = out_folder + "/" + tag + ".txt"
    with open(out_filename, "w") as f:
        start = 1 if start_time_at_1 else 0
        for t, tracks_t in enumerate(tracks, start):  # TODO this works?
            for track in tracks_t:
                if add_masks:
                    # MOTS methods
                    if track.class_ == 1:
                        print(t, track.track_id, track.class_,
                              *track.mask['size'], track.mask['counts'].decode(encoding='UTF-8'), file=f)
                    if track.class_ == 2:
                        print(t, track.track_id + 500, track.class_,
                              *track.mask['size'], track.mask['counts'].decode(encoding='UTF-8'), file=f)
                else:
                    # MOT methods
                    print(str(t+1) + "," + str(track.track_id)+","+str(track.box[0])+","+str(track.box[1])+","+str(track.box[2]-track.box[0])+","+str(track.box[3]-track.box[1])+","+str(track.score)+",-1,-1,-1", file=f)

                #print(t,track.track_id,track.box[0],track.box[1],track.box[2]-track.box[0], track.box[3]-track.box[1], track.score, -1, -1, -1, file=f)


def export_detections_for_sequence(tag, boxes, scores, reids, classes, masks, model_str, epoch, add_masks,
                                   out_folder=""):
    if out_folder == "":
        out_folder = "forwarded/" + model_str + "/detections/" + str(epoch)
    os.makedirs(out_folder, exist_ok=True)
    out_filename = out_folder + "/" + tag + ".txt"
    with open(out_filename, "w") as f:
        t = 0
        for boxes_t, scores_t, reids_t, classes_t, masks_t in zip(boxes, scores, reids, classes, masks):
            for box, score, reid, class_, mask in zip(boxes_t, scores_t, reids_t, classes_t, masks_t):
                if add_masks:
                    print(t, *box, score, class_, *mask['size'], mask['counts'].decode(encoding='UTF-8'), *reid, file=f)
                else:
                    print(t, *box, score, class_, *reid, file=f)
            t = t + 1


def import_detections_for_sequence(detections, add_masks=1, names=[]):
    mots_challenge_bias=0 #set it to 1 for mots challenge, and 0 for kitti mots. this is because images in motschallenge star from 1, t starts from 0

    out_filename = detections.args.det_dir + "/" + detections.videos[detections.args.vid] + ".txt"
    with open(out_filename) as f:
        content = f.readlines()
    boxes = []
    scores = []
    reids = []
    classes = []
    masks = []
    ts = []
    for line in content:
        entries = line.split(' ')
        # filter to classes_to_load if it is specified
        if detections.classes_to_load is not None and int(entries[6]) not in detections.classes_to_load:
            continue
        t = int(entries[0])
        if len(entries) <= 1:
            continue
        if names!=[] and t not in [int(n[0][:-4])-mots_challenge_bias for n in names]:
            continue
        if names!=[]: #for train (extracting a chunk, not the entire video
            start=int(names[0][0][:-4])-mots_challenge_bias
        else: #for test cases
            start=0
        while t + 1-start > len(boxes):
            boxes.append([])
            scores.append([])
            reids.append([])
            classes.append([])
            ts.append([])
            masks.append([])
        if entries[-1].endswith("\n"):
            entries[-1]=entries[-1][:-1]
        temp_mask=torch.tensor(cocomask.decode({'size': [int(entries[7]), int(entries[8])],
                                 'counts': entries[9].strip().encode(encoding='UTF-8')}))
        #print(t, temp_mask.shape, temp_mask.sum())
        if float(entries[5])>detections.args.det_thresh and temp_mask.sum()>10:
            if not detections.bb_format=="trackrcnn":
                boxes[t-start].append(torch.tensor([float(entries[1]),float(entries[2]),
                                                    float(entries[1])+float(entries[3]),
                                                    float(entries[2])+float(entries[4])]).cuda(detections.args.track_device))
            else: #trackrcnn bounding box format is xyxy
                boxes[t - start].append(torch.tensor([float(entries[1]), float(entries[2]),
                                                      float(entries[3]),
                                                      float(entries[4])]).cuda(detections.args.track_device))
            scores[t-start].append(torch.tensor(float(entries[5])).cuda(detections.args.track_device))
            classes[t-start].append(torch.tensor(int(entries[6])).cuda(detections.args.track_device))
            ts[t-start].append(int(entries[0])+mots_challenge_bias)
            reids[t - start].append(torch.tensor([float(e) for e in entries[10:]]).cuda(detections.args.track_device))
            if add_masks:
                temp_mask={'size': [int(entries[7]), int(entries[8])],
                                 'counts': entries[9].strip().encode(encoding='UTF-8')}
                masks[t-start].append(torch.tensor(cocomask.decode(temp_mask)))



            else:
                masks[t-start].append([])
                reids[t-start].append([float(e) for e in entries[7:]])

    diff=len(names)-len(boxes)
    for i in range(0,diff):
        boxes.append([])
        scores.append([])
        reids.append([])
        classes.append([])
        masks.append([])
        ts.append([])

    if names!=[] and mots_challenge_bias==0:
        for i in range(len(boxes)):
            if boxes[i]!=[]:
                boxes[i]=torch.stack(boxes[i])
                classes[i] = torch.stack(classes[i])
                reids[i]=torch.stack(reids[i])
                scores[i] = torch.stack(scores[i])
                masks[i] = torch.stack(masks[i])
            else:
                boxes[i] = torch.tensor(boxes[i])
                classes[i] = torch.tensor(classes[i])
                reids[i] = torch.tensor(reids[i])
                scores[i] = torch.tensor(scores[i])
                masks[i] = torch.tensor(masks[i])

    return boxes, scores, reids, classes, masks, ts

def import_detections_for_mot(tag, detections_import_path,
                                    device=0,names=[], bb_format="others", thresh=-2):
    mots_challenge_bias=0 #set it to 1 for mots challenge, and 0 for kitti mots. this is because images in motschallenge star from 1, t starts from 0

    out_filename = detections_import_path + "/" + tag + "/det/tracktor_prepr_det.txt"
    with open(out_filename) as f:
        content = f.readlines()
    boxes = []
    scores = []
    reids = []
    classes = []
    masks = []
    ts = []

    for line in content:
        entries = line.split(',')
        # filter to classes_to_load if it is specified
        t = int(entries[0])
        if len(entries) <= 1:
            continue
        if names!=[] and t not in [int(n[0][:-4])-mots_challenge_bias for n in names]:
            continue
        if names!=[]: #for train (extracting a chunk, not the entire video
            start=int(names[0][0][:-4])-mots_challenge_bias
        else: #for test cases
            start=1
        while t + 1-start > len(boxes):
            boxes.append([])
            scores.append([])
            reids.append([])
            classes.append([])
            ts.append([])
            masks.append([])
        if entries[-1].endswith("\n"):
            entries[-1]=entries[-1][:-1]

        #print(t, temp_mask.shape, temp_mask.sum())

        if float(entries[6])>thresh:
            if not bb_format=="trackrcnn":
                boxes[t-start].append(torch.tensor([float(entries[2]),float(entries[3]),
                                                    float(entries[2])+float(entries[4]),
                                                    float(entries[3])+float(entries[5])]).cuda(device))
            else: #trackrcnn bounding box format
                boxes[t - start].append(torch.tensor([float(entries[2]), float(entries[3]),
                                                      float(entries[4]),
                                                      float(entries[5])]).cuda(device))
            scores[t-start].append(torch.tensor(float(entries[6])).cuda(device))
            classes[t-start].append(torch.tensor([2]).cuda(device))
            ts[t-start].append(int(entries[0])+mots_challenge_bias)
            reids[t - start].append([])

            masks[t-start].append([])

    diff=len(names)-len(boxes)
    for i in range(0,diff):
        boxes.append([])
        scores.append([])
        reids.append([])
        classes.append([])
        masks.append([])
        ts.append([])

    if names!=[] and mots_challenge_bias==0:
        for i in range(len(boxes)):
            if boxes[i]!=[]:
                boxes[i]=torch.stack(boxes[i])
                classes[i] = torch.stack(classes[i])
                reids[i]=torch.tensor(reids[i])
                scores[i] = torch.stack(scores[i])
                masks[i] = torch.tensor(masks[i])
            else:
                boxes[i] = torch.tensor(boxes[i])
                classes[i] = torch.tensor(classes[i])
                reids[i] = torch.tensor(reids[i])
                scores[i] = torch.tensor(scores[i])
                masks[i] = torch.tensor(masks[i])


    return boxes, scores, reids, classes, masks, ts


def import_detections_for_mot_lift(tag, detections_import_path,
                              device=0, names=[], bb_format="others", thresh=0.85):
    mots_challenge_bias = 0  # set it to 1 for mots challenge, and 0 for kitti mots. this is because images in motschallenge star from 1, t starts from 0

    out_filename = detections_import_path + "/" + tag + "/det/det_preprocessed.csv"
    with open(out_filename) as f:
        content = f.readlines()
    boxes = []
    scores = []
    reids = []
    classes = []
    masks = []
    ts = []

    for line in content:

        entries = line.split(',')
        if entries[0]=="Frame":
            continue
        # filter to classes_to_load if it is specified
        t = int(entries[0])
        if len(entries) <= 1:
            continue
        if names != [] and t not in [int(n[0][:-4]) - mots_challenge_bias for n in names]:
            continue
        if names != []:  # for train (extracting a chunk, not the entire video
            start = int(names[0][0][:-4]) - mots_challenge_bias
        else:  # for test cases
            start = 1
        while t + 1 - start > len(boxes):
            boxes.append([])
            scores.append([])
            reids.append([])
            classes.append([])
            ts.append([])
            masks.append([])
        if entries[-1].endswith("\n"):
            entries[-1] = entries[-1][:-1]

        # print(t, temp_mask.shape, temp_mask.sum())

        if float(entries[5]) > thresh:
            if not bb_format == "trackrcnn":
                boxes[t - start].append(torch.tensor([float(entries[1]), float(entries[2]),
                                                      float(entries[1]) + float(entries[3]),
                                                      float(entries[2]) + float(entries[4])]).cuda(device))
            else:  # trackrcnn bounding box format
                boxes[t - start].append(torch.tensor([float(entries[1]), float(entries[2]),
                                                      float(entries[3]),
                                                      float(entries[4])]).cuda(device))
            scores[t - start].append(torch.tensor(float(entries[5])).cuda(device))
            classes[t - start].append(torch.tensor([2]).cuda(device))
            ts[t - start].append(int(entries[0]) + mots_challenge_bias)
            reids[t - start].append([])

            masks[t - start].append([])

    diff = len(names) - len(boxes)
    for i in range(0, diff):
        boxes.append([])
        scores.append([])
        reids.append([])
        classes.append([])
        masks.append([])
        ts.append([])

    if names != [] and mots_challenge_bias == 0:
        for i in range(len(boxes)):
            if boxes[i] != []:
                boxes[i] = torch.stack(boxes[i])
                classes[i] = torch.stack(classes[i])
                reids[i] = torch.tensor(reids[i])
                scores[i] = torch.stack(scores[i])
                masks[i] = torch.tensor(masks[i])
            else:
                boxes[i] = torch.tensor(boxes[i])
                classes[i] = torch.tensor(classes[i])
                reids[i] = torch.tensor(reids[i])
                scores[i] = torch.tensor(scores[i])
                masks[i] = torch.tensor(masks[i])


    return boxes, scores, reids, classes, masks, ts

def import_detections_for_sequence_MOTChallenge(tag, detections_import_path, model_str, epoch, add_masks,
                                                classes_to_load=None, device=0, trackrcnn=False):
    # for dataset MOTSChallenge the detections start from frame 0, but images start from 0001. this function is to deal with that discrepency.

    out_filename = detections_import_path + "/" + tag + ".txt"
    with open(out_filename) as f:
        content = f.readlines()
    boxes = []
    scores = []
    reids = []
    classes = []
    masks = []
    ts = []
    for line in content:
        entries = line.split(' ')
        # filter to classes_to_load if it is specified
        if classes_to_load is not None and int(entries[6]) not in classes_to_load:
            continue

        t = int(entries[0])
        print(t)
        if len(entries) <= 1:
            continue
        while t + 1 > len(boxes):
            boxes.append([])
            scores.append([])
            reids.append([])
            classes.append([])
            ts.append([])
            masks.append([])
        boxes[t].append(torch.tensor([float(entries[1]), float(entries[2]), float(entries[3]), float(entries[4])]))
        scores[t].append(torch.tensor(float(entries[5])))
        classes[t].append(torch.tensor(int(entries[6])))
        ts[t].append(t + 1)  # this is where the detection first frame is made 1 instead of 0.
        if add_masks:

            temp_mask = {'size': [int(entries[7]), int(entries[8])],
                         'counts': entries[9].strip().encode(encoding='UTF-8')}
            masks[t].append(torch.tensor(cocomask.decode(temp_mask)))

            reids[t].append(torch.tensor([float(e) for e in entries[10:]]))


        else:
            masks[t].append([])
            reids[t].append([float(e) for e in entries[7:]])
    # max_len_seq=100000
    # while max_len_seq > len(boxes):
    #   boxes.append([])
    #   scores.append([])
    #   reids.append([])
    #   classes.append([])
    #   masks.append([])
    #   ts.append([])

    # transform into numpy arrays
    for t in range(len(boxes)):
        if len(boxes[t]) > 0:
            boxes[t] = torch.stack(boxes[t])
            scores[t] = torch.stack(scores[t])
            classes[t] = torch.stack(classes[t])
            reids[t] = torch.stack(reids[t])
            ts[t] = np.array(ts[t])
    return boxes, scores, reids, classes, masks, ts


def load_optical_flow(tag, optical_flow_path):
    import pickle
    if os.path.exists(optical_flow_path + "/preprocessed_" + tag):
        with open(optical_flow_path + "/preprocessed_" + tag, 'rb') as input:
            flows = pickle.load(input)
    else:
        flow_files_x = sorted(glob.glob(optical_flow_path + "/" + tag + "/*_x_minimal*.png"))
        flow_files_y = sorted(glob.glob(optical_flow_path + "/" + tag + "/*_y_minimal*.png"))
        assert len(flow_files_x) == len(flow_files_y)
        flows = [open_flow_png_file([x, y]) for x, y in zip(flow_files_x, flow_files_y)]
        with open(optical_flow_path + "/preprocessed_" + tag, 'wb') as output:
            pickle.dump(flows, output, pickle.HIGHEST_PROTOCOL)
    return flows


def open_flow_png_file(file_path_list):
    # Funtion from Kilian Merkelbach.
    # Decode the information stored in the filename
    flow_png_info = {}
    assert len(file_path_list) == 2
    for file_path in file_path_list:
        file_token_list = os.path.splitext(file_path)[0].split("_")
        minimal_value = int(file_token_list[-1].replace("minimal", ""))
        flow_axis = file_token_list[-2]
        flow_png_info[flow_axis] = {'path': file_path,
                                    'minimal_value': minimal_value}

    # Open both files and add back the minimal value
    for axis, flow_info in flow_png_info.items():
        png_reader = png.Reader(filename=flow_info['path'])
        flow_2d = np.vstack(map(np.uint16, png_reader.asDirect()[2]))

        # Add the minimal value back
        flow_2d = flow_2d.astype(np.int16) + flow_info['minimal_value']

        flow_png_info[axis]['flow'] = flow_2d

    # Combine the flows
    flow_x = flow_png_info['x']['flow']
    flow_y = flow_png_info['y']['flow']
    flow = np.stack([flow_x, flow_y], 2)

    return flow


def get_images_optical_flow(path, seq):
    images = []

    dir_images = path + "images/"
    dirflow1 = path + "flow_skip0/dir1/"
    dirflow2 = path + "flow_skip0/dir2/"
    if not os.path.exists(dirflow1):
        os.mkdir(dirflow1)
    if not os.path.exists(dirflow2):
        os.mkdir(dirflow2)

    if not os.path.exists(dirflow1 + seq):
        os.mkdir(dirflow1 + seq)
    if not os.path.exists(dirflow2 + seq):
        os.mkdir(dirflow2 + seq)

    #############load images
    print("Appending Images...")
    image_files = sorted(os.listdir(dir_images + seq))
    for file in image_files:
        images.append(cv2.imread(dir_images + seq + "/" + file))

    #############load optical_flow

    optical_flow_skip0 = []

    print("Appending Optflow Images...")

    for im in range(1, len(images)):
        if os.path.exists(dirflow1 + seq + "/" + image_files[im][:-4] + ".npy"):
            fl_1 = np.load(dirflow1 + seq + "/" + image_files[im][:-4] + ".npy")
        else:
            print("calculating opt flow1..", im)
            fl_1, _ = get_optical_flow(images[im - 1], images[im])
            np.save(dirflow1 + seq + "/" + image_files[im][:-4] + ".npy", fl_1)

        # if os.path.exists(dirflow2 + seq + "/" + image_files[im][:-4]+".npy"):
        #     fl_2=np.load(dirflow2 + seq + "/" + image_files[im][:-4]+".npy")
        #
        # else:
        #     print("calculating opt flow1..",im)
        #     fl_2,_ = get_optical_flow(images[im], images[im - 1])
        #     np.save(dirflow2 + seq + "/" + image_files[im][:-4] + ".npy", fl_2)

        optical_flow_skip0.append(fl_1)  # @(fl_1 - fl_2) / 2)

    optical_flow_skip1 = []
    dirflow1 = path + "flow_skip1/dir1/"
    dirflow2 = path + "flow_skip1/dir2/"

    if not os.path.exists(dirflow1):
        os.mkdir(dirflow1)
    if not os.path.exists(dirflow2):
        os.mkdir(dirflow2)

    if not os.path.exists(dirflow1 + seq):
        os.mkdir(dirflow1 + seq)
    if not os.path.exists(dirflow2 + seq):
        os.mkdir(dirflow2 + seq)

    print("Appending Optflow 2 Images...")
    for im in range(2, len(images)):

        if os.path.exists(dirflow1 + seq + "/" + image_files[im][:-4] + ".npy"):
            fl_1 = np.load(dirflow1 + seq + "/" + image_files[im][:-4] + ".npy")
        else:
            print("calculating opt flow0..", im)
            fl_1, _ = get_optical_flow(images[im - 2], images[im])
            np.save(dirflow1 + seq + "/" + image_files[im][:-4] + ".npy", fl_1)

        # if os.path.exists(dirflow2 + seq + "/" + image_files[im][:-4] + ".npy"):
        #     fl_2 = np.load(dirflow2 + seq + "/" + image_files[im][:-4] + ".npy")
        #
        # else:
        #     print("calculating opt flow0..",im)
        #     fl_2,_ = get_optical_flow(images[im], images[im - 2])
        #     np.save(dirflow2 + seq + "/" + image_files[im][:-4] + ".npy", fl_2)

        optical_flow_skip1.append(fl_1)  # - fl_2) / 2)

    return images, optical_flow_skip0, optical_flow_skip1


def dataset_specifics_test(args):
    if "KITTI" in args.data_dir and args.dataset=="val" and args.mots:
        videos=["0002", "0006", "0007","0008","0010","0013", "0014","0016", "0018"]
        start_at_1=False
        #kitti_mots_val
    elif "KITTI" in args.data_dir and args.dataset=="test" and args.mots:
        videos=["0000", "0001", "0002","0003","0004","0005", "0006","0007", "0008", "0009", "0010", "0011","0012","0013","0014","0015", "0016","0017", "0018", "0019","0020", "0021","0022","0023","0024","0025", "0026","0027", "0028" ]
        #kitti_mots_test
        start_at_1=False
    elif "Challenge" in args.data_dir and args.dataset=="val" and args.mots:
        videos=["0002", "0005", "0009","0011"]
        #mots_challenge
        start_at_1=False
    elif "Challenge" in args.data_dir and args.dataset=="test" and args.mots:
        videos=["0001", "0006", "0007","0012"]
        #mots_challenge
        start_at_1=False

    elif "MOT17" in args.data_dir and args.dataset=="test" and not args.mots:
        tags=["01", "03", "06", "07", "08", "12", "14"]
        videos=["MOT17-"+tg+"-"+args.mot_method for tg in tags]
        #MOT17 test
        start_at_1=False

    elif "MOT17" in args.data_dir and args.dataset=="train" and not args.mots:
        tags=["02", "04", "05", "09", "10", "11", "13"]
        videos=["MOT17-"+tg+"-"+args.mot_method for tg in tags]
        #MOT17 train
        start_at_1=False
    elif not args.mots: #for mot20, 2Dmot2015
        videos=sorted(os.listdir(os.path.join(args.data_dir, args.dataset)))
        start_at_1=False

    return videos, start_at_1



def parameter_specifics_test(args, lmbda, track_device=0):


    #if not args.car:
        #lmbda=torch.tensor([1.1136,  8.1956,  2.0,  0.0000,  0.2011,  2.0275,  0.5, -0.1000]).cuda(track_device)
        #mrcnn, tests, #trackrcnn people, motschallenge
        #lmbda=torch.tensor([0.3329, 3.2097, 6.2284, 0.0000, 0.1441, 0.8352, 2.2944, 0.0000]).cuda(track_device)
            #pointtrack

    #else:
        #lmbda=torch.tensor([ 0.2260,  1.8769, 1.4456,  0.0, 0.0039,  0.6579, 0.1630,0.0]).cuda(track_device)
        #lmbda = torch.tensor([0.3250, 2.0900, 5.7378, 0.0000, 0.1398, 0.3441, 2.4856, 0.0000],requires_grad=True,device="cuda")
                #pointtrack

        #lmbda=torch.tensor([ 0.2260,  1.8769, 1.4456,  0.0, 0.0039,  0.6579, 0.1630,0.0]).cuda(track_device)
                #mrcnn, tests

        #lmbda=torch.tensor([0.5136,  5.1956,  5.4846,  0.0000,  0.4011,  0.6275,  3.5586, -0.1000]).cuda(track_device)
                #trackrcnn

    if not args.iou:
        lmbda[1]=0.
        lmbda[5]=0.
    if not args.app:
        lmbda[2]=0.
        lmbda[6]=0.
    if not args.dist:
        lmbda[0] = 0.
        lmbda[4] = 0.

    return lmbda
