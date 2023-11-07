import assignmentspace_mots.datasets.dataset as dataset
from assignmentspace_mots.models.mots_models import load_seg_model
import assignmentspace_mots.utils as utils

from assignmentspace_mots.detection_space import DetectionSpace, miniDetectionSpace
from assignmentspace_mots.assignment_space import AssignmentSpace


from configs import *

import torch.utils.data
from torch import optim
from torch.autograd import *
import numpy as np
import pickle
import os
import pdb
import time
import os.path
import sys
from types import SimpleNamespace
import cv2
#import interpolate

def train(itr):

    tic = time.clock()
    if not args.use_given_detections:
        model.train()

    for batch_idx, (images, gt_labels, vid, image_names) in enumerate(train_loader):
        # images: list of n images for mots, list of n image_paths for mot (since mot datasets are larger)
        # gt_labels: labels corresponding to the images

        print(vid[0],image_names)

        detections=DetectionSpace(args, image_names, vid[0], reid_model)
        detections.get_images(images)
        detections.get_optical_flow()
        detections.get_detections(names=image_names)
        # rearrange the ground truth to match the order of predictions
        detections.create_assignment_labels(gt_labels.copy(), predictions_mots,det_class_ids=[1,2], mots=args.mots)


        L_star = torch.tensor([0.]).cuda(args.track_device)
        L_GT = torch.tensor([0.]).cuda(args.track_device)

        if np.array([len(i) for i in boxes]).sum()>0:

            assignments=AssignmentSpace(args, detections, lmbda, train=True, parallelize=True)
            assignments.get_assignment_space(labels=tracking_labels)
            assignments.get_best_path()

        loss_seg =torch.tensor([0.]).cuda(args.seg_device1)

        if args.use_segloss:
            if args.seg_model_to_use=="MaskRCNN50":
                loss_seg=torch.stack([i[j] for i in seg_losses for j in i.keys()]).sum()
            else:
                loss_seg = 10*seg_losses
        loss_track=(L_GT-L_star)#+
        if args.use_segloss and (args.use_trackloss_theta or args.use_trackloss_lambda):
            loss=loss_track+loss_seg.cuda(args.track_device)
        if args.use_segloss and not (args.use_trackloss_theta or args.use_trackloss_lambda):
            loss=loss_seg
        if not args.use_segloss and (args.use_trackloss_theta or args.use_trackloss_lambda):
            loss=loss_track


        if epoch>0 and loss.requires_grad:
            if args.use_segloss or args.use_trackloss_theta:
                optimizer_theta.zero_grad()
            if args.use_trackloss_lambda:
                optimizer_lambda.zero_grad()

            loss.backward()
            if args.use_segloss or args.use_trackloss_theta:
                optimizer_theta.step()
            if args.use_trackloss_lambda:
                optimizer_lambda.step()


        if np.array([len(i) for i in boxes]).sum()>0:
            print("Epoch: ", epoch, " Itr: ", itr, ", y_star", y_star, " y_GT", y_GT, " L_star", L_star.item(), " L_GT", L_GT.item(), " delta_star",
                  delta_star, " lmbda", lmbda.detach())

        itr = itr + 1

        if np.array([len(i) for i in boxes]).sum() > 0:

            store["all_y_GT"].append(y_GT)
            store["all_y_star"].append(y_star)
            store["all_L_star"].append(L_star.item())
            store["all_L_GT"].append(L_GT.item())
            store["all_delta"].append(delta_star.item())
            store["tracking_loss"].append(delta_star.item() - L_star.item() + L_GT.item())

            if lmbda.grad!=None:
                store["all_grad_lmbda"].append([lmbda.grad[i].item() for i in range(0,len(lmbda))])
            else:
                store["all_grad_lmbda"].append([0. for i in range(0, len(lmbda))])
            store["all_lmbda"].append(lmbda.data.clone())

            store["mask_rcnn_loss"].append(loss_seg.data)

            store["all_iters"].append(itr)
            if not os.path.exists(outf_samples + "epoch" + str(epoch)):
                os.mkdir(outf_samples + "epoch" + str(epoch))
            if G!=[]:
                utils.plots_display.display_data(images_pic.cpu(), predictions_mots, y_star, y_GT, G,
                                    path=outf_samples + "epoch" + str(epoch) + "/"+vid[0]+"_delta"+str(int(delta_star.item()))+"_from" + image_names[0][0])
            del G


        if not args.use_given_detections:
            del predictions_mots, tracking_labels
            del loss, loss_track, loss_seg, seg_losses, L_GT, L_star
        del images, gt_labels,n_optflow_skip0, n_optflow_skip1
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated())
        #del track_grads

    #if epoch % 5 == 0:
    if not args.use_given_detections:
        torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (outf_model, epoch))
    else:
        torch.save(lmbda, '%s/lambda_epoch_%d.pth' % (outf_model, epoch))

    toc = time.clock()
    print("time", toc - tic)

    return itr


def test(lmbda):

    if args.car==True:
        caregory="car"
    else:
        category="person"


    os.makedirs(outf + "/test", exist_ok=True)
    os.makedirs(outf + "/test/" +category+ "/Instances_txt", exist_ok=True)
    if not args.mots:
        os.makedirs(outf + "/test/interp", exist_ok=True)
        os.makedirs(outf + "/test/interp/Instances_txt", exist_ok=True)

    tic = time.clock()
    if not args.use_given_detections:
        model.eval()
        theta = list(model.seg_model.parameters())
    else:
        theta = []

    lmbda=utils.file_helper.parameter_specifics_test(args, lmbda, track_device=args.track_device)

    videos, start_at_1=utils.file_helper.dataset_specifics_test(args)




    for batch_idx, (images, gt_labels, vid, image_names) in enumerate(val_loader):
        # images: list of n images
        # gt_labels: labels corresponding to the images

        print("*** Creating Detection Space ***")
        detections=DetectionSpace(args, vid[0], reid_model)
        print("Getting images...")
        detections.get_images(images)
        print("Getting optical flow...")
        detections.get_optical_flow(image_names)
        print("Getting detections...")
        detections.get_detections()


        L_star = torch.tensor([0.]).cuda(args.track_device)
        L_GT = torch.tensor([0.]).cuda(args.track_device)




        if np.array([len(i) for i in detections.boxes]).sum() > 0:
            print("*** Creating Assignment Space ***")
            assignments=AssignmentSpace(args, detections, lmbda, train=False, parallelize=False)
            print("Constructing the space...")
            assignments.get_assignment_space()
            print("Best path...")
            assignments.get_best_path()
            print("Get the tracks...")
            detections.get_tracks(assignments)



        all_tracks=utils.mots_helper.get_track_elements(detections, args)

        if args.mots:
            hyp_tracks =utils.mots_helper.make_disjoint(all_tracks, "score")
            utils.file_helper.export_tracking_result_in_kitti_format(vid[0], hyp_tracks, True, "",
                                                           out_folder=outf + "/test/"+category+"/Instances_txt",
                                                           start_time_at_1=start_at_1)
        else:
            utils.file_helper.export_tracking_result_in_kitti_format(vid[0], all_tracks, False, "",out_folder=outf + "/test/epoch" + str(args.epochs) + "/Instances_txt",start_time_at_1=start_at_1)
            interpolate.save_interpolated_tracks(outf + "/test/Instances_txt/",
                                                 outf + "/test/interp/Instances_txt/",vid[0]+".txt")
        print("done: ", vid, image_names)

        del assignments, detections, images
        torch.cuda.empty_cache()

    # plot_all_test(all_vids, vids_y_star,vids_y_GT,location+"/test/epoch"+str(epoch)+".png")
    return


args = eval(sys.argv[1]).get_args()
args = SimpleNamespace(**args)
if not args.train:
    args.vid=int(sys.argv[2])

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory':False}

# datasets
if args.mots:
    if args.train:
        mots_train_dict=dataset.get_mots_dict(args.data_dir + args.dataset)
        trainData = dataset.MOTS_dataset(mots_train_dict, args.n)
        train_loader = torch.utils.data.DataLoader(trainData, shuffle=True, **kwargs)
    else:
        mots_val_dict = dataset.get_mots_dict(args.data_dir + args.dataset, vid=args.vid)
        valData = dataset.MOTS_dataset(mots_val_dict, len(mots_val_dict), train=False)
        val_loader = torch.utils.data.DataLoader(valData, shuffle=True, **kwargs)  # my_collate

else: # evaluate on mot
    import structmots_datasets.mot_dataset as mot_dataset
    if args.train:
        mot_train_dict=mot_dataset.get_mot_dict(args.data_dir + args.dataset, method=args.mot_method)
        trainData = mot_dataset.MOT_dataset(mot_train_dict, args.n)
        train_loader = torch.utils.data.DataLoader(trainData, shuffle=True, **kwargs)
    else:
        mot_val_dict = mot_dataset.get_mot_dict(args.data_dir + args.dataset, vid=args.vid, method=args.mot_method)
        valData = mot_dataset.MOT_dataset(mot_val_dict, len(mot_val_dict), train=False)
        val_loader = torch.utils.data.DataLoader(valData, shuffle=True, **kwargs)  # my_collate



outf = args.outf+"/"#+folder


if not args.use_given_detections:
    model=load_seg_model(args.seg_model_path, args.reid_model_path, args.resume_lambda_model_path, which_model=args.seg_model_to_use,car=args.car, train=args.train, joint_train_load=args.joint_train_load)

    theta = list(model.seg_model.parameters())
    lmbda = model.tracking_head.weight[0]

    init_lmbda =model.tracking_head.weight.data[0]

else:
    model=[]
    theta=[]
    if not args.train:
        #lmbda = torch.load(args.track_model_path+"lambda_epoch_"+str(args.epochs)+".pth")
        lmbda = torch.tensor([6.93, 4.96, 0.47, 0.0000, -1.8, -0.45, 0.10, 0.0000], requires_grad=False, device="cuda")
        lmbda.requires_grad=False
    else:
        lmbda = torch.load(args.resume_lambda_model_path)

    init_lmbda = lmbda.clone()


if args.reid_model_path!=None:
    from assignmentspace_mots.models import reid_models
    reid_model = reid_models.REID_model(path=args.reid_model_path, car=args.car).cuda(args.track_device)

if args.train:
    params=[]
    if args.use_segloss or args.use_trackloss_theta:
        optimizer_theta = optim.SGD(list(model.seg_model.parameters()), lr=args.lr_theta, momentum=0.9)
    if args.use_trackloss_lambda:
        if model!=[]:
            optimizer_lambda = optim.SGD(list(model.tracking_head.parameters()), lr=args.lr_lambda, momentum=0.9)
        else:
            optimizer_lambda = optim.SGD([lmbda], lr=args.lr_lambda, momentum=0.9)


    outf_samples=outf +"/samples/"
    outf_model=outf + "/model_stored/"
    os.makedirs(outf, exist_ok = True)
    os.makedirs(outf_samples, exist_ok=True)
    os.makedirs(outf_model, exist_ok=True)

store={}
for i in args.storekeys:
    store[i]=[]

t_0=time.time()
itr_train=0
if args.train:
    for epoch in range(1, args.epochs+1):
        print("Train")
        print(torch.cuda.memory_allocated())
        itr_train= train(itr_train)

        store["epochs"].append(epoch)
        torch.cuda.empty_cache()

        utils.plots_display.plot_all(store, init_lmbda, lr_lmbda=args.lr_lambda, lr_theta=args.lr_theta,  location=outf)

if not args.train:
    test(lmbda)
    del model
    del val_loader
    del valData
    torch.cuda.empty_cache()
