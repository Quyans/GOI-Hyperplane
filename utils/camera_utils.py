#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch

import numpy as np
from scipy.spatial.transform import Rotation as R
import transformations as tfs

from scipy.interpolate import interp1d

import bisect

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if hasattr(cam_info, 'semantic') and cam_info.semantic is not None:
        # resized_semantic = torch.nn.functional.interpolate(cam_info.semantic.unsqueeze(0), size=resolution[::-1], mode='bilinear').squeeze(0)
        resized_semantic = cam_info.semantic
        semantic_path = cam_info.semantic_path
    else:
        resized_semantic = None
        semantic_path = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id,
                  semantic=resized_semantic, semantic_name=semantic_path,
                  data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry





# def interpolate_transform_matrices(matrices, num_samples, times=None):
#     """
#     对4x4变换矩阵进行插值,通过提取和插值四元数和平移向量。
#     插值是在每两个位姿之间进行的，并将所有的插值结果汇总为一个综合的结果。
    
#     :param matrices: 包含4x4变换矩阵的列表
#     :param times: 对应于每个位姿的时间戳列表
#     :param num_samples: 想要生成的插值样本数
#     :return: 插值后的4x4变换矩阵列表
#     """
    
#     if times is None:
#         times = np.linspace(0, len(matrices) - 1, len(matrices))  # Default time intervals

#     translations = np.array([mat[:3, 3] for mat in matrices])
#     quaternions = np.array([tfs.quaternion_from_matrix(mat) for mat in matrices])

#     interpolated_matrices = []

#     # 对每一对相邻位姿进行插值，并确保包含段的结束点
#     for i in range(len(matrices) - 1):
#         local_times = np.linspace(times[i], times[i+1], num_samples, endpoint=True)
#         local_translations = interp1d(times[i:i+2], translations[i:i+2], axis=0)(local_times)
#         local_quaternions = [tfs.quaternion_slerp(quaternions[i], quaternions[i+1], (t-times[i])/(times[i+1]-times[i])) for t in local_times]

#         # 创建插值后的矩阵
#         for trans, quat in zip(local_translations, local_quaternions):
#             mat = np.eye(4)
#             mat[:3, :3] = tfs.quaternion_matrix(quat)[:3, :3]
#             mat[:3, 3] = trans
#             interpolated_matrices.append(mat.astype(np.float32))

#     # 确保在每一对位姿的最后一个插值点被添加
#     if len(matrices) > 1:
#         final_trans = translations[-1]
#         final_quat = quaternions[-1]
#         final_mat = np.eye(4)
#         final_mat[:3, :3] = tfs.quaternion_matrix(final_quat)[:3, :3]
#         final_mat[:3, 3] = final_trans
#         interpolated_matrices.append(final_mat.astype(np.float32))
        
#     return interpolated_matrices



def interpolate_transform_matrices(matrices, num_samples, times=None):
    """
    对4x4变换矩阵进行插值,通过提取和插值四元数和平移向量。
    插值是在每两个位姿之间进行的，并将所有的插值结果汇总为一个综合的结果。
    
    :param matrices: 包含4x4变换矩阵的列表
    :param times: 对应于每个位姿的时间戳列表
    :param num_samples: 想要生成的插值样本数
    :return: 插值后的4x4变换矩阵列表
    """
    
    if times is None:
        times = np.linspace(0, len(matrices) - 1, len(matrices))  # Default time intervals

    translations = np.array([mat[:3, 3] for mat in matrices])
    quaternions = np.array([tfs.quaternion_from_matrix(mat) for mat in matrices])

    interpolated_matrices = []

    # 对每一对相邻位姿进行插值，并确保包含段的结束点
    for i in range(len(matrices) - 1):
        # 设置endpoint为False以避免在两段之间重复插值点
        endpoint = False if i < len(matrices) - 2 else True
        local_times = np.linspace(times[i], times[i+1], num_samples, endpoint=endpoint)
        local_translations = interp1d(times[i:i+2], translations[i:i+2], axis=0)(local_times)
        local_quaternions = [tfs.quaternion_slerp(quaternions[i], quaternions[i+1], (t-times[i])/(times[i+1]-times[i])) for t in local_times]

        # 创建插值后的矩阵
        for trans, quat in zip(local_translations, local_quaternions):
            mat = np.eye(4)
            mat[:3, :3] = tfs.quaternion_matrix(quat)[:3, :3]
            mat[:3, 3] = trans
            interpolated_matrices.append(mat.astype(np.float32))

    return interpolated_matrices
