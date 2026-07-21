import argparse
from pyskl.smp import *
from pyskl.models import build_model
import torch
from mmcv import Config
import copy as cp
from collections import OrderedDict

def getModel(cfg_path):
    model_cfg = Config.fromfile(cfg_path)
    model = build_model(model_cfg.model)

    return model


# The difference is in dim-1
def padding(weight, new_shape):
    new_weight = weight.new_zeros(new_shape)
    new_weight[:, :weight.shape[1]] = weight
    return new_weight

def merge(rgb_model, pose_model, output_path, cfg_path):
    rgb_ckpt = torch.load(rgb_model, map_location='cpu')['state_dict']
    pose_ckpt = torch.load(pose_model, map_location='cpu')['state_dict']

    rgb_ckpt = {k.replace('backbone', 'backbone.rgb_path').replace('fc_cls', 'fc_rgb'): v for k, v in rgb_ckpt.items()}
    pose_ckpt = {k.replace('backbone', 'backbone.pose_path').replace('fc_cls', 'fc_pose'): v for k, v in pose_ckpt.items()}

    old_ckpt = {}
    old_ckpt.update(rgb_ckpt)
    old_ckpt.update(pose_ckpt)

    ckpt = cp.deepcopy(old_ckpt)
    name = 'backbone.rgb_path.layer3.0.conv1.conv.weight'
    ckpt[name] = padding(ckpt[name], (256, 640, 3, 1, 1))
    name = 'backbone.rgb_path.layer3.0.downsample.conv.weight'
    ckpt[name] = padding(ckpt[name], (1024, 640, 1, 1, 1))
    name = 'backbone.rgb_path.layer4.0.conv1.conv.weight'
    ckpt[name] = padding(ckpt[name], (512, 1280, 3, 1, 1))
    name = 'backbone.rgb_path.layer4.0.downsample.conv.weight'
    ckpt[name] = padding(ckpt[name], (2048, 1280, 1, 1, 1))
    name = 'backbone.pose_path.layer2.0.conv1.conv.weight'
    ckpt[name] = padding(ckpt[name], (64, 160, 3, 1, 1))
    name = 'backbone.pose_path.layer2.0.downsample.conv.weight'
    ckpt[name] = padding(ckpt[name], (256, 160, 1, 1, 1))
    name = 'backbone.pose_path.layer3.0.conv1.conv.weight'
    ckpt[name] = padding(ckpt[name], (128, 320, 3, 1, 1))
    name = 'backbone.pose_path.layer3.0.downsample.conv.weight'
    ckpt[name] = padding(ckpt[name], (512, 320, 1, 1, 1))
    ckpt = OrderedDict(ckpt)

    out = os.path.join(output_path, 'rgbpose_conv3d_init_2.pth')
    torch.save({'state_dict': ckpt}, out)
    print(f"Merged checkpoint saved to: {out}")

    #validate that the model can load the merged checkpoint
    print("Validate merged checkpoint in model")
    model = getModel(cfg_path)
    model.load_state_dict(ckpt, strict=False)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge pose and rgb pretrained weights for joint RGB_Pose training')
    parser.add_argument('--rgb-model', required=True, help='File with the pretrained RGB model, e.g. rgb_only.pth')
    parser.add_argument('--pose-model', required=True, help='File with the pretrained POSE model, e.g. pose_only.pth')
    parser.add_argument('--config-path', default=None, help='Path to the config file holding the model definition to validate that the merged checkpoints can be loaded, e.g. rgbpose_conv3d.py')
    parser.add_argument('--output-path', required=True, help='Folder to save the merged checkpoint')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.config_path is not None:
        cfg_path = args.config_path
    else:
        cfg_path = 'julia/WLASL300/AblationStudies/2_RGBPose_Conv3d/rgbpose_conv3d.py'

    merge(args.rgb_model, args.pose_model, args.output_path, cfg_path)