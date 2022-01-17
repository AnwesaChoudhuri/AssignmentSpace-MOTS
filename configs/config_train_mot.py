
import copy
import torch



args = dict(
    train=True,
    mots=False, #true for mots, false for mot
    n=5,
    cuda=True,
    data_dir="data/MOT17/",
    dataset="train", #train, test, or val
    use_optical_flow=True,
    optical_flow_path="data/MOT17/train/",
    optical_flow_method="RAFT_optical_flow",

    
    epochs=40,
    seed=1,
    lr_theta=0.00001, #learning rate of theta (doesn't matter if use_segloss=False and use_trackloss_theta=False)
    lr_lambda=0.005, #learning rate of lambda

    outf='output/MOT17/train/tracking/FRCNN_withoptflow',
    det_dir="data/MOT_other_dets/MOT17_MPN", # doesn't matter if use_given_detections is false
    mot_method="FRCNN", #FRCNN, SDP or DPM

    second_order=True,
    real=True,
    quadratic=True,
    iou_type="bb",
    
    use_segloss=False,
    use_trackloss_theta=False,
    use_trackloss_lambda=True,

    seg_device0=0, 
    seg_device1=0, 
    track_device=0,
    reid_resnet=True,

    use_given_detections=True,
    seg_model_to_use="SpatialEmbedding", # Choose from ("MaskRCNN50, "SpatialEmbedding"). Only useful if use_given_detections=False.
    seg_model_path='PointTrack/pointTrack_weights/best_seed_model.pthCar',
    car=False, #False for person
    reid_model_path='PointTrack/person_finetune_tracking/checkpoint.pth',
                    #"PointTrack/car_finetune_tracking/checkpoint.pth",
    resume_lambda_model_path= "output/KITTI_MOTS/train/tracking/pointtrack/losses_tracklambda/model_stored/lambda_epoch_20.pth",

    storekeys=("all_y_star", "all_y_GT", 
        "all_L_star", "all_L_GT", "all_delta", 
        "all_grad_lmbda", "all_lmbda" , "all_iters",
        "epochs","tracking_loss", "class_loss", "box_reg", "mask_loss", 
        "mask_rcnn_loss","track_grad_theta_norm", "mrcnn_grad_theta_norm"))

    
def get_args():
    return copy.deepcopy(args)



