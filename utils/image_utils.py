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
import os, sys
import cv2
import torch
from torchvision import transforms

import matplotlib
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def apply_mask(A, mask):
    # 检查mask的维度并根据A的维度进行调整
    if mask.dim() == 1:
        # 扩展mask以匹配A的第一个维度
        mask = mask.view(-1, *((1,) * (A.dim() - 1)))
    
    # 扩展mask以匹配A的全部维度
    mask = mask.expand_as(A)
    return mask

def compute_mask_ratio(refer_mask, mask):
    if refer_mask.any()==False:
        return 0
    
    # refer_mask【H,W,1】 torch
    # 计算两个mask的交集占refer_mask的比例
    # matching_elements = torch.eq(refer_mask, mask)
    intersection = torch.logical_and(refer_mask, mask)
    # 计算 maskA 中的非零元素数量
    nonzero_elements_in_refer_mask = torch.count_nonzero(refer_mask)
    # 计算匹配元素在 maskA 中的比例
    proportion = torch.count_nonzero(intersection) / nonzero_elements_in_refer_mask
    return proportion.item()

def calculate_target_iou(target_mask, mask):
    # 计算两个mask的交集占target_mask的比例
    intersection = torch.logical_and(target_mask, mask)
    # 计算 maskA 中的非零元素数量
    nonzero_elements_in_target_mask = torch.count_nonzero(target_mask)
    # 计算匹配元素在 maskA 中的比例
    proportion = torch.count_nonzero(intersection) / nonzero_elements_in_target_mask
    return proportion.item()

def calculate_iou(label, pred):

    pred_inds = pred == 1
    label_inds = label == 1
    intersection = torch.logical_and(pred_inds, label_inds).sum()
    union = torch.logical_or(pred_inds, label_inds).sum()
    if union == 0:
        iou = float('nan')  # 避免除以零
    else:
        iou = float(intersection) / float(max(union, 1))

    return iou

# mPA 计算的是图像中被正确分类的像素数与总像素数的比例。这通常在图像分割任务中使用，其中每个像素都被分类为某个类别。它的平均值是对所有类别的平均像素准确率。
def calculate_mean_pixel_accuracy(true_labels, predicted_labels):
    
    # 确保输入的维度和形状是相同的
    assert true_labels.shape == predicted_labels.shape

    # 计算每个类别的像素准确率
    accuracy_class_1 = torch.sum((predicted_labels == 1) & (true_labels == 1)).float() / torch.sum(true_labels == 1).float()
    accuracy_class_0 = torch.sum((predicted_labels == 0) & (true_labels == 0)).float() / torch.sum(true_labels == 0).float()

    # 防止除以零的情况
    accuracy_class_1 = accuracy_class_1 if torch.sum(true_labels == 1) > 0 else torch.tensor(0.)
    accuracy_class_0 = accuracy_class_0 if torch.sum(true_labels == 0) > 0 else torch.tensor(0.)

    # 计算平均像素准确率
    mPA = (accuracy_class_1 + accuracy_class_0) / 2
    return mPA


# 
def calculate_mean_precision(true_labels, predicted_labels):
    # 确保输入的维度和形状是相同的
    assert true_labels.shape == predicted_labels.shape

    # 计算每个类别的precision
    precision_class_1 = torch.sum((predicted_labels == 1) & (true_labels == 1)).float() / torch.sum(predicted_labels == 1).float()
    precision_class_0 = torch.sum((predicted_labels == 0) & (true_labels == 0)).float() / torch.sum(predicted_labels == 0).float()

    # 计算平均precision
    mP = (precision_class_1 + precision_class_0) / 2
    return mP

def transform_Image_to_tensor(image):
    # 将图像转换为张量
    transform = transforms.ToTensor()
    output_image = transform(image)
    return output_image

def generate_video(img_list, save_path):
        # 定义编解码器并创建VideoWriter对象
    output_path = os.path.join(save_path, 'video.mp4')  # 输出视频路径
    # frame_width, frame_height = 1920, 1080  # 修改为你的图片分辨率

    first_image = cv2.imread(img_list[0])
    height, width, layers = first_image.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10,
                            (width, height))

    # 读取每个文件并写入到视频中
    for img in img_list:
        image = cv2.imread(img)
        out.write(image)  # 将帧写入视频

    # 完成后释放所有资源
    out.release()
    print("save video path", save_path)


def clip_color(cos_sim, height, width, thresh=0.7, res_finetuned=False, coloring=False, device='cuda'):
    # 着色方案不一样
    if res_finetuned:
        non_zero_elements = cos_sim[cos_sim != 0]
        if len(non_zero_elements) == 0:
            min_non_zero = 0
        else:
            min_non_zero = torch.min(non_zero_elements)
        # rel = torch.clamp((cos_sim - min_non_zero) / (cos_sim.max() - min_non_zero + 0.1), 0, 1)
        rel = torch.clamp(cos_sim + 0.2, 0.1, 0.9)
    else:
        rel = torch.clamp((cos_sim - thresh - 0.05) / (cos_sim.max() - thresh), 0, 1)  # rel = torch.clamp((cos_sim - 0.47) / (0.49 - 0.47), 0, 1)  ##CLIP-only

    cmap = matplotlib.colormaps.get_cmap("turbo")

    heat_img = cmap(rel.detach().cpu().numpy()).astype(np.float32)
    heat_img = torch.from_numpy(heat_img).to(device)
    masked_hi = heat_img * cos_sim.unsqueeze(1) * 0.9
    masked_hi[cos_sim == 0] = 1
    masked_hi[:, 3] = 1
    if not coloring or res_finetuned:
        masked_hi[cos_sim != 0] = 0
    masked_hi = masked_hi.reshape(height, width, 4)

    masked_hi = (masked_hi.contiguous().clamp(
        0, 1).contiguous().detach().cpu().numpy())
    return masked_hi
