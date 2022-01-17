
import copy
import torch



args = dict(
    train=False,
    mots=False, #true for mots, false for mot
    n=5,
    cuda=True,
    data_dir="data/MOTChallenge/MOT17/",
    dataset="train", #train, test, or val
    use_optical_flow=True,
    optical_flow_path="data/MOTChallenge/MOT17/train/",
    optical_flow_method="RAFT_optical_flow",
    vid=1, # will read from 2nd argument
    reid_resnet=True,

    epochs=5, #which epoch to test
    seed=1,
    lr_theta=0.00001, #learning rate of theta
    lr_lambda=0.02, #learning rate of lambda

    outf='output/MOT17/train/tracking/FRCNN_withoptflow',
    det_dir="data/MOTChallenge/MOT_other_dets/MOT17_MPN",  # doesn't matter if use_given_detections is false
    mot_method="FRCNN",  # FRCNN, SDP or DPM

    second_order=True,
    real=True,
    quadratic=True,
    iou=True,
    iou_type="bb",
    app=True,
    dist=True,
    appthresh=0.95,
    leaf=1,
    leaf_dist=0, #how many leaf distances to compare (0 by default: checks just the best leaf)
    K=10,

    car=False,
    det_thresh=-2, # 0.7 for cars, 0.85 for pedestrians


    use_segloss=False,
    use_trackloss_theta=False, 
    use_trackloss_lambda=True, # useful because we want to know which directory to go to

    seg_device0=0, 
    seg_device1=0, 
    track_device=0,


    use_given_detections=True,
    save_files_only=False,
    seg_model_to_use="SpatialEmbedding", # Doesn't matter
    seg_model_path='PointTrack/pointTrack_weights/best_seed_model.pthCar', # Doesn't matter
    
    
    reid_model_path="PointTrack/person_finetune_tracking_boundingbox/checkpoint.pth",
                   #'PointTrack/car_finetune_tracking/checkpoint.pth'
    track_model_path="output/MOT17/train/tracking/FRCNN_withoptflow/losses_tracklambda/model_stored/",

    storekeys=("all_y_star", "all_y_GT", 
        "all_L_star", "all_L_GT", "all_delta", 
        "all_grad_lmbda", "all_lmbda" , "all_iters",
        "epochs","tracking_loss", "class_loss", "box_reg", "mask_loss", 
        "mask_rcnn_loss","track_grad_theta_norm", "mrcnn_grad_theta_norm"))

  
def get_args():
    return copy.deepcopy(args)




