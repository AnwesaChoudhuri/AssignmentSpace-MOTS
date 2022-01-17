
import pdb
def load_seg_model(seg_model_path, reid_model_path, lambda_path, which_model="SpatialEmbedding", car=True, train=True, joint_train_load=False):
    if which_model=="MaskRCNN50":
        existing_cfg, my_cfg = my_setup(args, folder=folder)
        import structmots_models.maskrcnn as mots_model

        model = mots_model.MRCNN_model(existing_cfg, my_cfg, mrcnn_device0=seg_device0, mrcnn_device1=seg_device1,
                                      track_device=track_device)
        for p in model.seg_model.parameters():
            p.requires_grad = False
        # for p in model.mrcnn_model.roi_heads.mask_head.mask_fcn8.parameters():
        #     p.requires_grad = True
        if train:
            for p in model.seg_model.roi_heads.mask_head.deconv.parameters():
                p.requires_grad = True
            for p in model.seg_model.roi_heads.mask_head.predictor.parameters():
                p.requires_grad = True

            for p in model.seg_model.roi_heads.box_predictor.parameters():
                p.requires_grad = True



    if which_model=="SpatialEmbedding":
        import structmots_models.se as mots_model

        model = mots_model.SE_model(seg_model_path, reid_model_path,lambda_path=lambda_path, car=car, joint_train_load=joint_train_load)

        for p in model.seg_model.parameters():
            p.requires_grad = False
        if train:

            for p in model.seg_model.module.decoders[1].output_conv.parameters():
                p.requires_grad = True
            # for p in model.seg_model.module.decoders[1].parameters():
            #     p.requires_grad = True
            for p in model.seg_model.module.decoders[1].layers._modules['4'].parameters():
                p.requires_grad = True
            for p in model.seg_model.module.decoders[1].layers._modules['5'].parameters():
                p.requires_grad = True


            for p in model.seg_model.module.decoders[0].output_conv.parameters():
                p.requires_grad = True
            # for p in model.seg_model.module.decoders[0].parameters():
            #     p.requires_grad = True
            for p in model.seg_model.module.decoders[0].layers._modules['4'].parameters():
                p.requires_grad = True
            for p in model.seg_model.module.decoders[0].layers._modules['5'].parameters():
                p.requires_grad = True

    return model
