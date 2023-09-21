_base_ = ['../_base_/default_runtime.py']
encoder_config_file = 'transformer_utils/configs/smpler_x/encoder/body_encoder_huge.py'
encoder_pretrained_model_path = '../pretrained_models/vitpose_huge.pth'

use_adversarial_train = True
import os
import os.path as osp

# will be update in exp
num_gpus = -1
exp_name = 'output/exp1/pre_analysis'

# quick access
save_epoch = 1
lr = 1e-5
min_lr = 5e-7
end_epoch = 5
train_batch_size = 16

syncbn = True
bbox_ratio = 1.2

input_body_shape = (256, 192)
princpt = (input_body_shape[1] / 2, input_body_shape[0] / 2)
model = dict(
    type='SMPLer_X',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
    ),
    focal = (5000, 5000)
    input_img_shape = (512, 384),
    input_body_shape = input_body_shape,
    input_hand_shape = (256, 256),
    input_face_shape = (192, 192),
    output_hm_shape = (16, 16, 12),
    output_hand_hm_shape = (16, 16, 16),
    output_face_hm_shape = (8, 8, 8),
    testset = 'EHF',
    princpt = princpt,
    feat_dim = 1280,
    upscale = 4,
)
# continue
continue_train = True
pretrained_model_path = '../path_to_smpler_x_h32/snapshot.pth.tar'

# dataset setting
agora_fix_betas = True
agora_fix_global_orient_transl = True
agora_valid_root_pose = True

# for ubody ft
dataset_list = ['Human36M', 'MSCOCO', 'MPII', 'AGORA', 'EHF', 'SynBody', 'GTA_Human2', \
    'EgoBody_Egocentric', 'EgoBody_Kinect', 'UBody', 'PW3D', 'MuCo', 'PROX']
trainset_3d = ['PW3D']
trainset_2d = []
trainset_humandata = ['GTA_Human2', 'EgoBody_Kinect', 'InstaVariety', 'HumanSC3D']


use_cache = True

# strategy
data_strategy = 'balance' # 'balance' need to define total_data_len
total_data_len = 1000000 # assign number or 'auto' for concat length

Talkshow_train_sample_interval = 10

# fine-tune
fine_tune = None # 'backbone', 'head', None for full network tuning

smplx_loss_weight = 1.0 #2 for agora_model for smplx shape
smplx_pose_weight = 10.0

smplx_kps_3d_weight = 100.0
smplx_kps_2d_weight = 1.0
net_kps_2d_weight = 1.0

agora_benchmark = 'agora_model' # 'agora_model', 'test_only'

model_type = 'smpler_x_h'
\


## =====FIXED ARGS============================================================
## model setting
hand_pos_joint_num = 20
face_pos_joint_num = 72
num_task_token = 24
num_noise_sample = 0

## UBody setting
train_sample_interval = 10
test_sample_interval = 100
make_same_len = False

## input, output size

body_3d_size = 2
hand_3d_size = 0.3
face_3d_size = 0.3
camera_3d_size = 2.5

## training config
print_iters = 100
lr_mult = 1

## testing config
test_batch_size = 32

## others
num_thread = 4
vis = False

## directory
output_dir, model_dir, vis_dir, log_dir, result_dir, code_dir = None, None, None, None, None, None
