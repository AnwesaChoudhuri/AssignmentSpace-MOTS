
import copy
import torch



args = dict(
    train=False,
    mots=True, #true for mots, false for mot
    n=5,
    cuda=True,
    data_dir="data/KITTI_MOTS/",
    dataset="test", #train, test, or val
    optical_flow_path="data/KITTI_MOTS/test/RAFT_optical_flow/",
    use_optical_flow=True,
    vid=1,

    epochs=5, #which epoch to test
    seed=1,
    lr_theta=0.00001, #learning rate of theta
    lr_lambda=0.02, #learning rate of lambda

    resume_lambda_model_path= None, # lambda has to be retrieved from the seg model path itself, so it is None
    
    outf='output/KITTI_MOTS/test/tracking/pointtrack',
    det_dir="output/KITTI_MOTS/test/detections/detections_pointtrack",


    second_order=True,
    real=True,
    quadratic=True,
    iou=True,
    iou_type="mask",
    app=True,
    dist=True,
    appthresh=0.95,
    leaf=1,
    leaf_dist=0, #how many leaf distances to compare (0 by default: checks just the best leaf)
    K=10,

    car=True,
    det_thresh=0.7, # 0.7 for cars, 0.85 for pedestrians


    use_segloss=True,
    use_trackloss_theta=True,
    use_trackloss_lambda=True,

    seg_device0=0, 
    seg_device1=0, 
    track_device=0,


    use_given_detections=False,
    save_files_only=False,
    seg_model_to_use="SpatialEmbedding", # Choose from ("MaskRCNN50, "SpatialEmbedding"). Only useful if use_given_detections=False.
    seg_model_path="output/KITTI_MOTS/train/tracking/pointtrack/losses_seg_tracktheta_tracklambda/model_stored/model_epoch_5.pth",
    joint_train_load=True, 


    reid_model_path="PointTrack/car_finetune_tracking/checkpoint.pth",
                   #'PointTrack/person_finetune_tracking/checkpoint.pth'
    storekeys=("all_y_star", "all_y_GT", 
        "all_L_star", "all_L_GT", "all_delta", 
        "all_grad_lmbda", "all_lmbda" , "all_iters",
        "epochs","tracking_loss", "class_loss", "box_reg", "mask_loss", 
        "mask_rcnn_loss","track_grad_theta_norm", "mrcnn_grad_theta_norm"))

  
def get_args():
    return copy.deepcopy(args)




