import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cam_utils import intrinsic_to_fov
import torchvision
# from .utils import get_rays, safe_normalize, extract_bbox_np
from PIL import Image
from cam_utils import visualize_poses, visualize_poses_stabledreamfusion

DIR_COLORS = np.array([
    [255, 0, 0, 255], # front
    [0, 255, 0, 255], # side
    [0, 0, 255, 255], # back
    [255, 255, 0, 255], # side
    [255, 0, 255, 255], # overhead
    [0, 255, 255, 255], # bottom
], dtype=np.uint8)



def get_view_direction(thetas, phis, overhead, front):
    #                   phis: [B,];          thetas: [B,]
    # front = 0             [-front/2, front/2)
    # side (cam left) = 1   [front/2, 180-front/2)
    # back = 2              [180-front/2, 180+front/2)
    # side (cam right) = 3  [180+front/2, 360-front/2)
    # top = 4               [0, overhead]
    # bottom = 5            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    phis = phis % (2 * np.pi)
    res[(phis < front / 2) | (phis >= 2 * np.pi - front / 2)] = 0
    res[(phis >= front / 2) & (phis < np.pi - front / 2)] = 1
    res[(phis >= np.pi - front / 2) & (phis < np.pi + front / 2)] = 2
    res[(phis >= np.pi + front / 2) & (phis < 2 * np.pi - front / 2)] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


class NeRFDataset:
    def __init__(self, opt, device, type='train'):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        
        self.preload = opt.preload # preload data into GPU

        # self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        # self.offset = opt.offset # camera offset
        # self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        # self.fp16 = opt.fp16 # if preload, load into fp16.
        
        # self.size = size
        # self.root_path = opt.path
        self.training = self.type in ['train', 'all']

        # self.near = self.opt.min_near
        # self.far = 500 # infinite
        # self.num_rays = self.opt.num_rays if self.training else -1
        # self.rand_pose = opt.rand_pose

        self.mode = "scannet"
        
        folder_path_dir = '/home/fchao/users/daishaohui/datasets/scannet/scans/scene0241_02/image'
        color_path_dir = os.path.join(folder_path_dir, 'color')
        pose_path_dir = os.path.join(folder_path_dir, 'pose')
        intrinsic_path = os.path.join(folder_path_dir, 'intrinsic',"intrinsic_color.txt")
        semantic_pred_path_dir = os.path.join(folder_path_dir, 'semantic_predict')
        
        select_step = 3
        if self.mode=="scannet":
            
            # load image size
            image_w = 1296.0
            image_h = 968.0
            if self.opt.H:
                self.H = (float)(self.opt.H)
                self.W = (float)(self.opt.W)
            else:
                self.H = 512.0
                self.W = 512.0
            
            self.downscale = image_h / self.opt.H
            target_h = (int)(opt.H)
            target_w = (int)(image_w / self.downscale)
            # target_h = int(opt.H)
            # target_w =  int(image_w // downscale)
            
            # read intrinsics
            intrinsic_data = np.loadtxt(intrinsic_path)
            fx = intrinsic_data[0,0] / self.downscale
            fy = intrinsic_data[1,1] / self.downscale
            
            cx = intrinsic_data[0,2] / self.downscale
            cy = intrinsic_data[1,2] / self.downscale
            
            # center crop， 对于fov来说 不需要考虑cx和cy的问题，centercrop不影响f_x f_y
            h_crop = (image_h - self.H)/2
            w_crop = (image_w - self.W)/2    
            cx_new = cx - w_crop
            cy_new = cy - h_crop
            center_crop_transform = torchvision.transforms.CenterCrop((self.H, self.W))
            self.intrinsics = np.array([fx, fy, cx_new, cy_new])
            
            self.fov_x, self.fov_y = intrinsic_to_fov(self.intrinsics[0], self.intrinsics[1], self.W, self.H)
            # read poses
            
            self.poses = []
            self.images = []
            self.semantic_preds = []
            
            pose_names = os.listdir(pose_path_dir)
            selected_files = pose_names[::select_step]
            
            center_crop_transform = torchvision.transforms.CenterCrop((self.H, self.W))

            for f in tqdm.tqdm(selected_files, desc=f'Loading {self.mode} data'):
                
                file_name = f.split("/")[-1].split(".")[0]
                pose_path = os.path.join(pose_path_dir,f)
                rgb_path = os.path.join(color_path_dir, file_name + ".jpg")
                # semantic_pred_path = os.path.join(semantic_pred_path_dir, file_name + ".png")
                # if file_name == '1365':
                #     print(455)
                # load pose，  scannet的pose是c2w，换为opengl的坐标系
                pose = np.loadtxt(pose_path).astype(np.float32)
                if True in np.isnan(pose) or True in np.isinf(pose):
                    continue
                # # 旋转部分：转置3x3旋转矩阵
                # rot_transpose = pose[:3, :3].T
                # # 平移部分：将平移向量乘以-1后左乘旋转矩阵的转置
                # trans = -np.dot(rot_transpose, pose[:3, 3])
                # # 构建w2c矩阵
                # inv_matrix = np.identity(4)
                # inv_matrix[:3, :3] = rot_transpose
                # inv_matrix[:3, 3] = trans
                # pose = inv_matrix
                pose[:3, 1:3] *= -1

                # rgb image
                color_rgb = Image.open(rgb_path)
                color_rgb = resize_image_pil(color_rgb, target_h, target_w)
                color_rgb = center_crop_transform(color_rgb)
                color_rgb = np.array(color_rgb)
                color_rgb = color_rgb.astype(np.float32) / 255
                
                # semantic_rgb = Image.open(semantic_pred_path)
                # semantic_rgb = resize_image_pil(semantic_rgb, target_h, target_w)
                # semantic_rgb = center_crop_transform(semantic_rgb)
                # semantic_rgb = np.array(semantic_rgb)
                # semantic_rgb = semantic_rgb.astype(np.float32) / 255
                # semantic_rgb = semantic_rgb.astype(np.float32)
                # semantic_rgb = resize_image(semantic_rgb, self.H, self.W)
                # semantic_rgb = image.astype(np.float32) / 255 # [H, W, 3/4]
                
                self.images.append(color_rgb)
                self.poses.append(pose)
                # self.semantic_preds.append(semantic_rgb)

        
        self.poses = np.stack(self.poses, axis=0)
        self.poses = torch.from_numpy(self.poses) # [N, 4, 4]
        visualize_poses(self.poses.numpy(), size=0.1, file_name="our.glb")

        
        if len(self.images)!= 0:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, 3]
        if len(self.semantic_preds)!= 0:
            self.semantic_preds = torch.from_numpy(np.stack(self.semantic_preds, axis=0)) # [N, H, W, 3]

        if self.preload: #True
            self.poses = self.poses.to(self.device)
            self.images = self.images.to(self.device)
            self.semantic_preds = self.semantic_preds.to(self.device)
    

    def collate(self, index):

        B = len(index)

        # poses = self.poses[index].to(self.device) # [B, 4, 4   
        poses = self.poses[index] # [B, 4, 4        
             
        intrinsics = self.intrinsics
        height = self.H
        width = self.W
        
        results = {
            'poses': poses,
            'H': height,
            'W': width,
            'fov_x':self.fov_x,
            'fov_y':self.fov_y
        }

        if len(self.semantic_preds) != 0:
            semantic_preds = self.semantic_preds[index].to(self.device) #[B,H,W,3]
        results['semantic_preds'] = semantic_preds
        
        if len(self.images) != 0:
            images = self.images[index].to(self.device) #[B,H,W,3]
        results["images"] = images
        
        if hasattr(self,"frames_name"):
            frames_name = self.frames_name[index] # [B, 1] str
            results['frames_name'] = frames_name

        return results
    @property
    def get_camera_info(self):
        return [self.H, self.W, self.fov_x, self.fov_y]

    # def dataloader(self, batch_size=None):
    #     batch_size = batch_size or self.opt.batch_size
    #     loader = DataLoader(list(range(self.size)), batch_size=batch_size, collate_fn=self.collate, shuffle=self.training, num_workers=0)
    #     loader._data = self
    #     return loader
    
    def dataloader(self,batch_size=None):
        batch_size = batch_size or self.opt.batch_size
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=batch_size, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        return loader, self.get_camera_info
    
    
# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

def nerf_matrix_to_nerf(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[0,0],pose[0,1],pose[0,2],pose[0,3] * scale + offset[0]],
        [pose[1,0],pose[1,1],pose[1,2],pose[1,3] * scale + offset[0]],
        [pose[2,0],pose[2,1],pose[2,2],pose[2,3] * scale + offset[0]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

def resize_image(img, Height, Width):
    if img.shape[0] != Height or img.shape[1] != Width:
        return cv2.resize(img, (Width, Height), interpolation=cv2.INTER_LINEAR)
    return img

def resize_image_pil(img, height, width):
    if img.size != (width, height):
        return img.resize((width, height), Image.BILINEAR)
    return img