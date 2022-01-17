
import copy
import torch



args = dict(
    train=True,
    mots=True, #true for mots, false for mot
    n=4,
    cuda=True,
    data_dir="data/KITTI_MOTS/",
    dataset="train", #train, test, or val
    optical_flow_path="data/KITTI_MOTS/train/RAFT_optical_flow/",
    use_optical_flow=True,
    
    epochs=4,
    seed=1,
    lr_theta=0.000001, #learning rate of theta
    lr_lambda=0.0005, #learning rate of lambda

    resume_lambda_model_path= "output/KITTI_MOTS/train/tracking/pointtrack/losses_tracklambda/model_stored/lambda_epoch_20.pth",

    outf='output/KITTI_MOTS/train/tracking/pointtrack',
    det_dir="output/KITTI_MOTS/train/detections/detections_pointtrack", # doesn't matter if use_given_detections is false

    second_order=True,
    real=True,
    quadratic=True,
    iou_type="mask",
    
    use_segloss=True,
    use_trackloss_theta=True,
    use_trackloss_lambda=True,

    seg_device0=0, 
    seg_device1=0, 
    track_device=0,

    use_given_detections=False,
    seg_model_to_use="SpatialEmbedding", # Choose from ("MaskRCNN50, "SpatialEmbedding"). Only useful if use_given_detections=False.
    seg_model_path='PointTrack/pointTrack_weights/best_seed_model.pthCar',
    car=True, #False for person
    reid_model_path="PointTrack/car_finetune_tracking/checkpoint.pth",
                   #'PointTrack/person_finetune_tracking/checkpoint.pth'
    joint_train_load=False, #resume from previous joint train model

    storekeys=("all_y_star", "all_y_GT", 
        "all_L_star", "all_L_GT", "all_delta", 
        "all_grad_lmbda", "all_lmbda" , "all_iters",
        "epochs","tracking_loss", "class_loss", "box_reg", "mask_loss", 
        "mask_rcnn_loss","track_grad_theta_norm", "mrcnn_grad_theta_norm"))

    
def get_args():
    return copy.deepcopy(args)



