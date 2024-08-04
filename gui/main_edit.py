import os, sys
import cv2
import time

from tqdm import tqdm
import numpy as np
import dearpygui.dearpygui as dpg
import matplotlib
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import rembg
from pathlib import Path

from scipy.spatial.transform import Rotation as R

import clip
from cam_utils import orbit_camera, OrbitCamera, get_c2w_with_RT
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize
from provider import NeRFDataset
from scene.dataset_readers_no_semantic import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.image_utils import apply_mask, compute_mask_ratio, calculate_iou, clip_color
from utils.graphics_utils import computeBBox
import argparse
from PIL import Image

from sklearn.cluster import DBSCAN
from torchvision.utils import save_image
import glob

# ext_path = os.path.join(os.path.dirname(__file__), 'ext')
# sys.path.append(ext_path)


sys.path.append("./ext")
sys.path.append("./ext/GroundingDINO")

from ext import EVA02CLIP, VisionLanguageAlign
from networks import LinearSVM, ConvergenceTracker

SEM_DIM = 10


class GUI:

    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui  # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.render_resolution = 512

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.buffer_sem = np.zeros((self.W, self.H, SEM_DIM), dtype=np.float32)
        # self.clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)
        self.need_update = True  # update buffer_image

        self._3d_edit = False
        self.retrieval = False
        self.nograd_mask = None
        self.move = None

        self.setting_pose = -1

        # workspace
        self.workspace = os.path.join(self.opt.outdir, self.opt.save_path)

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None
        self.model_lang = None

        self.enable_sd = True
        self.enable_sdxl = False
        self.enable_lods = False
        self.enable_zero123 = False
        self.enable_rgb = True

        self.strength = 0.99

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5
        self.clip_feature_thresh = 0.86
        self.clip_update = False

        # input text
        self.prompt = ""
        self.negative_prompt = ""
        self.clip_prompt = ''
        self.save_view_name = "current_view"

        self.clip_feature = None
        self.clip_feature_bias = None
        self.clip_feature_manual_bias = None

        self.res_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None

        self.epoch = 0
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        self.fp16 = False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.scheduler_update_every_step = False

        # data saver
        self.all_cameras = []
        self.relative_cameras = []
        self.gui_poses = []

        self.camera_select_step = 1

        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints":
            [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)

        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt
        if self.opt.target_prompt is not None:
            self.clip_prompt = self.opt.target_prompt
            # self.text_clip_encode(self.clip_prompt)
            # self.clip_feature = model_lang.forward_text([self.clip_prompt])["last_hidden_state_eot"]
            self.encode_text(self.clip_prompt)
        if self.opt.target_res_prompt is not None:
            self.res_prompt = self.opt.target_res_prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            if hasattr(self.opt, "semantic_MLP_load"
                       ) and self.opt.semantic_MLP_load is not None:
                self.renderer.initialize(input=self.opt.load,
                                         input_MLP=self.opt.semantic_MLP_load)
            else:
                self.renderer.initialize(self.opt.load)
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        self.resMLP = None
        self.res_finetuned = False

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    @torch.no_grad()
    def compute_clip_cos_sim(self, semantic_feature):
        compute_batch_size = 1000000  #100W
        cos_sim = []

        for i in range(0, semantic_feature.size(0), compute_batch_size):
            # 获取小批量数据
            sem_feature_batch = semantic_feature[i:i + compute_batch_size]
            sem_label_batch = self.renderer.MLP(sem_feature_batch)[:, :300]
            if self.renderer.LUT is not None:
                sem_label_batch = torch.softmax(sem_label_batch * 10,
                                                dim=-1).argmax(dim=-1)
                sem_label_batch = self.renderer.LUT[sem_label_batch]
            sem_label_batch = sem_label_batch / sem_label_batch.norm(
                dim=-1, keepdim=True)
            # print(sem_label.shape, self.clip_feature.shape)

            if self.res_finetuned:
                # 使用res finetune的MLP作为分割效果
                logit = (self.resMLP(sem_label_batch.cuda())).squeeze()
                print(logit.max(), logit.min())
                cos_sim_batch = (logit).sigmoid().squeeze(-1)
                clip_thresh = 0.5
            else:
                logit, manual_bias = self.cls_embd.compute_dot_product_logit_betweenTandI_manualbias(
                    sem_label_batch.cuda(), self.clip_feature.cuda())
                if self.clip_feature_manual_bias == None:
                    self.clip_feature_manual_bias = manual_bias
                cos_sim_batch = (logit).sigmoid().squeeze(-1)
                clip_thresh = self.clip_feature_thresh

            cos_sim_batch[cos_sim_batch < clip_thresh] = 0

            # 保存计算结果
            cos_sim.append(cos_sim_batch)
        cos_sim = torch.cat(cos_sim, dim=0)

        return cos_sim

    @torch.no_grad()
    def set_clip_mask(self):
        cos_sim = self.compute_clip_cos_sim(self.buffer_sem)
        t = self.clip_feature_thresh
        # rel = torch.clamp((cos_sim - t) / (cos_sim.max()-t), 0, 1) #归一化到[0,1] 用于展示

        masked_hi = clip_color(cos_sim,
                               height=self.H,
                               width=self.W,
                               res_finetuned=self.res_finetuned,
                               device=self.device)
        colormap, transparency = masked_hi[:, :, :3], masked_hi[:, :, 3:]
        self.buffer_image = (colormap * transparency * self.overlay_input_img_ratio +
                             self.buffer_image * (1 - transparency * self.overlay_input_img_ratio)).clip(0, 1)
        self.need_update = False

    @torch.no_grad()
    def set_clip_mask_with_mask_area(self):

        out = self.render_once(width=self.W, height=self.H)
        buffer_image = out[self.mode]  # [3, H, W]

        buffer_image = F.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        self.buffer_image = (buffer_image.permute(1, 2, 0).contiguous().clamp(
            0, 1).contiguous().detach().cpu().numpy())
        self.buffer_sem = out['semantics'].permute(1, 2,
                                                   0).detach().cuda().reshape(
                                                       -1, SEM_DIM)

        cos_sim = self.compute_clip_cos_sim(self.buffer_sem)

        t = self.clip_feature_thresh
        # rel = torch.clamp((cos_sim - t) / (cos_sim.max()-t), 0, 1) #归一化到[0,1] 用于展示
        rel = torch.clamp((cos_sim - 0.75) / (cos_sim.max() - 0.7), 0, 1)

        cmap = matplotlib.colormaps.get_cmap("turbo")

        heat_img = cmap(rel.detach().cpu().numpy())[:, :3].astype(np.float32)

        heat_img = torch.from_numpy(heat_img).reshape(self.H, self.W, 3).cuda()
        cos_sim = cos_sim.reshape(self.H, self.W).unsqueeze(2)
        masked_hi = heat_img * cos_sim * 0.9

        masked_hi = (masked_hi.contiguous().clamp(
            0, 1).contiguous().detach().cpu().numpy())

        # self.buffer_image = (
        #         masked_hi +
        #         self.buffer_image
        # ).clip(0, 1)

        target_mask = (cos_sim > 0).cpu().numpy()
        self.buffer_image = (
            self.buffer_image * (1 - target_mask) +
            (masked_hi * self.overlay_input_img_ratio + self.buffer_image *
             (1 - self.overlay_input_img_ratio)) * target_mask).clip(0, 1)

        dpg.set_value(
            "_texture",
            self.buffer_image)  # buffer must be contiguous, else seg fault!
        print("set_clip_mask with area")

        self.need_update = False

    def compute_gs_nograd_mask(self):
        gs_semantics = self.renderer.gaussians.get_semantics
        croped_semantic = self.compute_clip_cos_sim(gs_semantics)
        nograd_mask = croped_semantic == 0  #[N]
        nograd_mask = nograd_mask.to(croped_semantic.device)
        return nograd_mask

    @torch.no_grad()
    def pre_compute_relative_cameras(self):
        print("precompute...")
        self.renderer.gaussians.set_semantic_masks()
        self.need_update = True
        # 将相关的3d GS球mask出来并且设置 (1-mask) 的GS require_grad=False

        self.nograd_mask = self.compute_gs_nograd_mask()

        # self.renderer.gaussians.change_gs_learnable_feature_use_mask(self.nograd_mask)

        # 相关视角最大的相关像素数量
        max_relative_number = 0
        min_relative_ratio = 0.1
        self.relative_cameras = []

        for ind in tqdm(range(len(self.all_cameras))):
            camera = self.all_cameras[ind]

            C2W = get_c2w_with_RT(camera.R, camera.T)

            cur_cam = MiniCam(C2W, self.render_resolution,
                              self.render_resolution, self.cam.fovy,
                              self.cam.fovx, self.cam.near, self.cam.far)

            out = self.renderer.render(cur_cam)  #[3,H,W]
            out_semantic = out['semantics'].permute(1, 2, 0).detach().reshape(
                -1, SEM_DIM)
            cos_sim = self.compute_clip_cos_sim(out_semantic)

            if cos_sim.any():
                relative_pixel_number = torch.count_nonzero(cos_sim)
                max_relative_number = max(max_relative_number,
                                          relative_pixel_number)
                camera.relative_pixel_number = relative_pixel_number

                # semantic = out["semantics"].detach().permute(1, 2, 0).reshape(-1, SEM_DIM) #[HW, 10]
                # semantic_cropped = self.compute_clip_cos_sim(semantic)
                # semantic_mask = semantic_cropped > 0

                semantic_mask = cos_sim > 0
                semantic_mask = semantic_mask.to(cos_sim.device)
                semantic_mask = semantic_mask.reshape(
                    self.render_resolution, self.render_resolution,
                    -1).permute(2, 0, 1)  # [1,512,512]

                semantic_mask_dilated = semantic_mask.detach().to(
                    dtype=torch.float32).cpu().numpy().squeeze(0)
                dilate_kernel = 3
                dilate_iterations = 5
                mask_dilated = cv2.dilate(semantic_mask_dilated,
                                          np.ones(
                                              (dilate_kernel, dilate_kernel),
                                              np.uint8),
                                          iterations=dilate_iterations)

                mask_dilated = mask_dilated >= 0.5

                camera.semantic_mask = semantic_mask.detach()
                camera.semantic_mask_dilated = torch.from_numpy(
                    mask_dilated).unsqueeze(0).to(semantic_mask.device)

                self.relative_cameras.append(camera)
                counter = len(self.relative_cameras)

        # 处理一下所有的relative_cameras，如果他的relative_pixel_number小于max_relative_number * min_relative_ratio，就删除掉

        print("删除前相关相机数目:", len(self.relative_cameras))
        i = 0
        while i < len(self.relative_cameras):
            if self.relative_cameras[
                    i].relative_pixel_number < max_relative_number * min_relative_ratio:
                self.relative_cameras.remove(self.relative_cameras[i])
            else:
                i += 1
        print("删除后relative cameras:", len(self.relative_cameras))

    def clear_noralative_gs_grad(self, gs, nograd_mask):
        if gs._features_dc.grad is not None:
            # nograd_mask_expand = nograd_mask.expand_as(gs._features_dc.grad)
            nograd_mask_expand = apply_mask(gs._features_dc.grad, nograd_mask)
            gs._features_dc.grad[nograd_mask_expand == 1] = 0
        if gs._features_rest.grad is not None:
            # nograd_mask_expand = nograd_mask.expand_as(gs._features_rest.grad)
            nograd_mask_expand = apply_mask(gs._features_rest.grad,
                                            nograd_mask)
            gs._features_rest.grad[nograd_mask_expand == 1] = 0

        if gs._xyz.grad is not None:
            # nograd_mask_expand = nograd_mask.expand_as(gs._xyz.grad)
            nograd_mask_expand = apply_mask(gs._xyz.grad, nograd_mask)
            gs._xyz.grad[nograd_mask_expand == 1] = 0

        if gs._opacity.grad is not None:
            # nograd_mask_expand = nograd_mask.expand_as(gs._opacity.grad)
            nograd_mask_expand = apply_mask(gs._opacity.grad, nograd_mask)
            gs._opacity.grad[nograd_mask_expand == 1] = 0

        if gs._scaling.grad is not None:
            # nograd_mask_expand = nograd_mask.expand_as(gs._scaling.grad)
            nograd_mask_expand = apply_mask(gs._scaling.grad, nograd_mask)
            gs._scaling.grad[nograd_mask_expand == 1] = 0

        if gs._rotation.grad is not None:
            # nograd_mask_expand = nograd_mask.expand_as(gs._rotation.grad)
            nograd_mask_expand = apply_mask(gs._rotation.grad, nograd_mask)
            gs._rotation.grad[nograd_mask_expand == 1] = 0

        if gs._semantics.grad is not None:
            # nograd_mask_expand = nograd_mask.expand_as(gs._semantics.grad)
            nograd_mask_expand = apply_mask(gs._semantics.grad, nograd_mask)
            gs._semantics.grad[nograd_mask_expand == 1] = 0

    @torch.no_grad()
    def edit_delete(self):
        gs_semantics = self.renderer.gaussians.get_semantics
        croped_semantic = self.compute_clip_cos_sim(gs_semantics)

        crop_mask = croped_semantic > 0
        crop_mask = crop_mask.to(croped_semantic.device)
        self.renderer.gaussians.prune_points(crop_mask)
        self.need_update = True

    @torch.no_grad()
    def edit_retrieve(self):
        self.nograd_mask = self.compute_gs_nograd_mask()
        self.retrieval = True
        self.need_update = True

    def render_once(self,
                    width,
                    height,
                    camera=None,
                    gaussain_scale_factor=None):

        if camera is None:
            cur_cam = MiniCam(
                self.cam.pose,
                width,
                height,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )
        else:
            cur_cam = MiniCam(
                camera.pose,
                width,
                height,
                camera.fovy,
                camera.fovx,
                camera.near,
                camera.far,
            )

        if gaussain_scale_factor is None:
            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)
        else:
            out = self.renderer.render(cur_cam, gaussain_scale_factor)
        return out

    def train_epoch(self):

        self.epoch += 1
        self.step = 0

        bs = self.opt.batch_size
        self.total_iters_perepoch = len(self.relative_cameras) // bs
        # for ind in tqdm(range(len(self.relative_cameras))): #每个epoch都要遍历一遍所有的相关视角
        for i in tqdm(range(0, len(self.relative_cameras), bs),
                      total=self.total_iters_perepoch):
            current_cameras = self.relative_cameras[i:i + bs]
            if not self.training:
                break
            if self.setting_pose != -1:
                self.cam.import_pose(self.gui_poses[self.setting_pose])
                self.need_update = True
                self.setting_pose = -1

            self.train_step(cameras=current_cameras)
            self.need_update = True
            self.test_step()

            if self.gui:
                dpg.render_dearpygui_frame()

    def train_step(self, cameras):

        self.set_clip_mask_with_mask_area()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if self.opt.save_guidance and self.step % self.opt.save_guidance_interval == 0:
            save_guidance_folder = self.save_guidance_folder
            guidance_img_path = os.path.join(
                self.save_guidance_folder,
                "rgb_pred" + str(self.step) + ".png")
            # guidance_img_path = os.path.join(self.save_guidance_folder,"rgb_pred.png")
        else:
            save_guidance_folder = None
            guidance_img_path = None

        self.step += 1

        # update lr
        self.renderer.gaussians.update_learning_rate(self.step)

        loss = 0

        images = []
        semantic_masks = []
        semantic_masks_ori = []
        depths = []
        images_inpainted = []

        step_ratio = min(1, (self.total_iters_perepoch *
                             (self.epoch - 1) + self.step) /
                         (self.total_iters_perepoch * self.opt.max_epochs))

        # visualize_poses(poses)

        for index in range(len(cameras)):
            # image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
            # images.append(image)
            camera = cameras[index]
            pose = get_c2w_with_RT(camera.R, camera.T)
            # radii.append(radius)
            # distance=0 represents in origin

            cur_cam = MiniCam(pose, self.render_resolution,
                              self.render_resolution, self.cam.fovy,
                              self.cam.fovx, self.cam.near, self.cam.far)

            # bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
            bg_color = torch.tensor([1, 1, 1],
                                    dtype=torch.float32,
                                    device="cuda")
            out = self.renderer.render(cur_cam, bg_color=bg_color)

            image = out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]

            semantic_mask = camera.semantic_mask_dilated.unsqueeze(0)
            semantic_mask_ori = camera.semantic_mask.unsqueeze(0)

            images.append(image)
            semantic_masks.append(semantic_mask)
            semantic_masks_ori.append(semantic_mask_ori)

        images = torch.cat(images, dim=0)
        semantic_masks = torch.cat(semantic_masks, dim=0)
        semantic_masks_ori = torch.cat(semantic_masks_ori, dim=0)

        # guidance loss
        if self.enable_sd:
            # strength = step_ratio * 0.15 + 0.8
            if self.enable_sdxl:
                # prepare prompt
                pos_prompt = [self.prompt] * self.opt.batch_size
                neg_prompt = [self.negative_prompt] * self.opt.batch_size
                # use sd refine
                images_inpainted = self.guidance_sd.inpaint(
                    images,
                    semantic_masks,
                    pos_prompt,
                    neg_prompt,
                    guidance_scale=self.opt.sd_guidance_scale,
                    strength=self.strength,
                    guidance_img_path=guidance_img_path).float()
                images_inpainted = F.interpolate(
                    images_inpainted,
                    (self.render_resolution, self.render_resolution),
                    mode="bilinear",
                    align_corners=False)

                # refined_images = refined_images * mask
                mse_loss = nn.MSELoss(reduction='none')  #能手动应用mask
                sd_refine_loss = mse_loss(images, images_inpainted)
                # semantic_masks [1,1,512,512] 而 rgb是[1,3,512,512] 因此需要扩张
                semantic_masks_expanded = semantic_masks.expand_as(
                    sd_refine_loss)
                masked_sd_refine_loss = sd_refine_loss * semantic_masks_expanded
                sd_refine_loss = masked_sd_refine_loss.sum()

                loss = loss + self.opt.lambda_sd * sd_refine_loss
            else:
                inpainting_switch = True
                if inpainting_switch:
                    if self.enable_lods:
                        loss_sds, loss_embedding = self.guidance_sd.train_step(
                            images,
                            semantic_masks,
                            masks_nodilated=semantic_masks_ori,
                            guidance_scale=self.opt.sd_guidance_scale,
                            step_ratio=step_ratio,
                            save_guidance_path=guidance_img_path)
                        loss = loss + self.opt.lambda_sd * loss_sds + loss_embedding
                    else:
                        loss_sds = self.guidance_sd.train_step(
                            images,
                            semantic_masks,
                            masks_nodilated=semantic_masks_ori,
                            guidance_scale=self.opt.sd_guidance_scale,
                            step_ratio=step_ratio,
                            save_guidance_path=guidance_img_path)
                        loss = loss + self.opt.lambda_sd * loss_sds
                else:
                    guidance_scale = self.opt.sd_guidance_scale

                    # use sd refine
                    refined_images = self.guidance_sd.refine(
                        images,
                        guidance_scale=guidance_scale,
                        strength=self.strength,
                        guidance_img_path=guidance_img_path).float()
                    refined_images = F.interpolate(
                        refined_images,
                        (self.render_resolution, self.render_resolution),
                        mode="bilinear",
                        align_corners=False)

                    # refined_images = refined_images * mask
                    mse_loss = nn.MSELoss(reduction='none')  #能手动应用mask
                    sd_refine_loss = mse_loss(images, refined_images)
                    # semantic_masks [1,1,512,512] 而 rgb是[1,3,512,512] 因此需要扩张
                    semantic_masks_expanded = semantic_masks.expand_as(
                        sd_refine_loss)
                    masked_sd_refine_loss = sd_refine_loss * semantic_masks_expanded
                    sd_refine_loss = masked_sd_refine_loss.sum()

                    loss = loss + self.opt.lambda_sd * sd_refine_loss

                # loss = loss + self.opt.lambda_sd * self.guidance_sd.refine_step(images, guidance_scale=guidance_scale, step_ratio=step_ratio)

        # if self.enable_opacity_regular:
        #     # self.renderer.gaussians.get_opacity 已经过了sigmoid了
        #     # opacity max是1，均值大概0.3左右
        #     loss = loss + self.opt.lambda_opacity_regular * self.renderer.gaussians.get_opacity.mean()

        # optimize step
        if not loss:

            self.optimizer.zero_grad()
            return

        loss.backward()

        # 在3d上做no grad
        self.clear_noralative_gs_grad(self.renderer.gaussians,
                                      self.nograd_mask)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.enable_lods:
            optimizer_embeddings = self.guidance_sd.get_embedding_optimizer()
            optimizer_embeddings.step()
            optimizer_embeddings.zero_grad()

        # densify and prune
        # cur_step = self.total_iters_perepoch*((self.epoch-1)) + self.step
        # if cur_step >= self.opt.density_start_iter and cur_step <= self.opt.density_end_iter:
        #     viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
        #     self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        #     self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        #     if cur_step % self.opt.densification_interval == 0 :
        #         self.opt.max_num_pts=3000000
        #         self.opt.densify_grad_threshold==0.01
        #         if (self.renderer.gaussians.get_xyz.shape[0]<self.opt.max_num_pts):

        #             num_before = self.renderer.gaussians.get_xyz.shape[0]
        #             self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
        #             self.nograd_mask = self.compute_gs_nograd_mask()
        #             print("before densify and prune: ", num_before, " after: ", self.renderer.gaussians.get_xyz.shape[0])
        #         else:
        #             num_before = self.renderer.gaussians.get_xyz.shape[0]
        #             # self.renderer.gaussians.prune(min_opacity=0.01, extent=4, max_screen_size=1)
        #             self.renderer.gaussians.prune(min_opacity=0.01, extent=4, max_screen_size=0) #no extent
        #             print("before only prune: ", num_before, " after: ", self.renderer.gaussians.get_xyz.shape[0])

        #     if cur_step % self.opt.opacity_reset_interval == 0:
        #         num_before = self.renderer.gaussians.get_xyz.shape[0]
        #         self.renderer.gaussians.reset_opacity()
        #         self.nograd_mask = self.compute_gs_nograd_mask()
        #         print("before opacity reset: ", num_before, " after: ", self.renderer.gaussians.get_xyz.shape[0])

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                # f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
                f"epoch={self.epoch:5d} step={self.step: 5d}(+{self.train_steps: 2d}) loss={loss.item():.4f} gsnumber:{self.renderer.gaussians.get_xyz.shape[0]}",
            )

    @torch.no_grad()
    def test_step(self):

        # ignore if no need to update
        if not self.need_update:
            return
        # print("test step")
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image
            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            mask_ret = ~self.nograd_mask if self.retrieval else None

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor, gaussian_mask=mask_ret)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (
                        buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (buffer_image.permute(
                1, 2,
                0).contiguous().clamp(0,
                                      1).contiguous().detach().cpu().numpy())
            self.buffer_sem = out['semantics'].permute(
                1, 2, 0).detach().cuda().reshape(-1, SEM_DIM)

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio) +
                    self.input_img * self.overlay_input_img_ratio)

            self.need_update = False
            if (self.clip_prompt != '') and (self.model_lang is not None):
                self.clip_update = True
                # 在这里save一下 渲染的语义图和clip_feature
                # self.renderer.gaussians.set_semantic_masks()
                self.set_clip_mask()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time",
                          f"{t:.4f}ms ({int(1000 / t)} FPS)")
            dpg.set_value("_texture", self.buffer_image
                          )  # buffer must be contiguous, else seg fault!

    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024, savename=None):

        if mode == 'geo':
            path = os.path.join(self.workspace,
                                self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(
                path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(
                self.workspace,
                self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(
                path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3),
                                 device=self.device,
                                 dtype=torch.float32)
            cnt = torch.zeros((h, w, 1),
                              device=self.device,
                              dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui
                                                 or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )

                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                # import kiui
                # kiui.vis.plot_image(rgbs)

                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(
                    self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(
                    np.float32)).to(self.device)

                v_cam = torch.matmul(
                    F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0),
                    torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(
                    glctx, v_clip, mesh.f,
                    (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast,
                                          mesh.f)  # [1, H, W, 1]
                depth = depth.squeeze(0)  # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast,
                                        mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(
                    mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()

                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h,
                    w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )

                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1,
                                   algorithm="kd_tree").fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(
                search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            if savename:
                path = os.path.join(self.workspace,
                                    savename + 'point_cloud.ply')
            else:
                path = os.path.join(self.workspace, 'point_cloud.ply')
            self.renderer.gaussians.save_ply(path)

        config_path = os.path.join(self.workspace, 'config.txt')
        # 打开文件并写入对象的属性
        with open(config_path, 'w', encoding='utf-8') as file:
            for key, value in self.opt._content.items():
                file.write(f"{key}: {value}\n")

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
                tag="_primary_window",
                width=self.W,
                height=self.H,
                pos=[0, 0],
                no_move=True,
                no_title_bar=True,
                no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
                label="Control",
                tag="_control_window",
                width=600,
                height=self.H,
                pos=[self.W, 0],
                no_move=True,
                no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,
                                        (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,
                                        (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            def change_save_path(sender, app_data):
                self.workspace = os.path.join(self.opt.outdir, app_data)
                print("self.workspace:", self.workspace)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                def callback_update_clip_mask(sender, app_data):
                    self.clip_prompt = app_data
                    # self.text_clip_encode(self.clip_prompt)
                    # _text_features = self.clip_pretrained.encode_text(clip.tokenize([self.clip_prompt]).cuda()).type(torch.float32)
                    # self.clip_feature = (_text_features / _text_features.norm(dim=-1, keepdim=True)).cuda()
                    # self.clip_feature = model_lang.forward_text([self.clip_prompt])["last_hidden_state_eot"]
                    self.encode_text(self.clip_prompt)
                    self.need_update = True

                dpg.add_input_text(
                    label="CLIP",
                    default_value=self.clip_prompt,
                    on_enter=True,
                    callback=callback_update_clip_mask,
                )

                def callback_update_res_prompt(sender, app_data):
                    self.res_prompt = app_data

                dpg.add_input_text(
                    label="RES",
                    default_value=self.res_prompt,
                    # on_enter=True,
                    callback=callback_update_res_prompt,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                        directory_selector=False,
                        show=False,
                        callback=callback_select_input,
                        file_count=1,
                        tag="file_dialog_tag",
                        width=700,
                        height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")

                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.2f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                def callback_set_clip_feature_thresh(sender, app_data):
                    self.clip_feature_thresh = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="feature crop thresh",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.clip_feature_thresh,
                    callback=callback_set_clip_feature_thresh,
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex",
                                        theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=change_save_path,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button

                with dpg.group(horizontal=True):

                    dpg.add_button(label="init",
                                   tag="_button_init",
                                   callback=self.load_models)
                    dpg.bind_item_theme("_button_init", theme_button)

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # def callback_group_points(sender, app_data):

                    #     # tem_prompt = "sofa chair"
                    #     # 在这里save一下mask
                    #     res_mask, _ = self.pred_res_mask(self.res_prompt)
                    #     res_mask = res_mask.to(self.device)
                    #     self.group_points(res_mask=res_mask)

                    # dpg.add_text("grouping: ")
                    # dpg.add_button(
                    #     label="group", callback=callback_group_points
                    # )

                    def callback_save_current_view(sender, app_data):
                        # res_mask, pred_image = self.guidance_res.predict_res_mask(out_image.cpu(), prompt)
                        # mask_out_expand = res_mask.expand_as(pred_image)
                        # viz_images = torch.cat([pred_image.unsqueeze(0) ,mask_out_expand.unsqueeze(0)],dim=0)
                        # save_image(viz_images, "testfolder/"+ prompt + ".png")

                        os.makedirs(self.workspace, exist_ok=True)

                        tensor_img = torch.from_numpy(
                            self.buffer_image).permute(2, 0, 1)
                        save_path = os.path.join(self.workspace,
                                                 self.save_view_name + ".png")
                        save_image(tensor_img, save_path)
                        print("saved view in:", save_path)

                    dpg.add_text("saveview: ")
                    dpg.add_button(label="save",
                                   callback=callback_save_current_view)
                    dpg.add_input_text(
                        # label="",
                        default_value=self.save_view_name,
                        callback=callback_setattr,
                        user_data="save_view_name",
                    )

                with dpg.group(horizontal=True):

                    def callback_res_locate(sender, app_data):
                        res_mask, _ = self.pred_res_mask(
                            self.res_prompt, self.W, self.H)
                        res_mask = res_mask.to(self.device)
                        self.finetune_prompt_with_res(res_mask=res_mask)
                        self.need_update = True

                    def callback_refresh_res(sender, app_data):
                        self.resMLP = None
                        self.res_finetuned = False
                        self.need_update = True

                    dpg.add_text("RES:")
                    dpg.add_button(label="res_loc",
                                   callback=callback_res_locate)

                    dpg.add_button(label="refresh",
                                   callback=callback_refresh_res)

                    def callback_show_res_results(sender, app_data):
                        res_mask, _ = self.pred_res_mask(
                            self.res_prompt, self.W, self.H)
                        buffer_image = res_mask.repeat(3, 1, 1)
                        buffer_image = F.interpolate(
                            buffer_image.unsqueeze(0),
                            size=(self.H, self.W),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)

                        self.buffer_image = (buffer_image.permute(
                            1, 2, 0).contiguous().clamp(
                                0, 1).contiguous().detach().cpu().numpy())

                        dpg.set_value(
                            "_texture", self.buffer_image
                        )  # buffer must be contiguous, else seg fault!
                        print("pred res mask finushed")

                    dpg.add_button(label="show_res_masks",
                                   callback=callback_show_res_results)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # edit stuff
            with dpg.collapsing_header(label="Edit", default_open=False):
                # prompt stuff
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )
                # lr and train button
                with dpg.group(horizontal=True):

                    dpg.add_text("change Object: ")

                    def callback_precompute(sender, app_data):
                        self.pre_compute_relative_cameras()

                    dpg.add_button(label="Precompute",
                                   callback=callback_precompute)
                    # dpg.bind_item_theme("_button_train", theme_button)

                    dpg.add_button(label="start",
                                   tag="_button_train",
                                   callback=callback_train)
                    dpg.bind_item_theme("_button_train", theme_button)
                with dpg.group(horizontal=True):
                    dpg.add_text("Edit option: ")

                    def callback_delete(sender, app_data):
                        self.edit_delete()

                    dpg.add_button(label="delete",
                                   tag="_button_delete",
                                   callback=callback_delete)
                    dpg.bind_item_theme("_button_delete", theme_button)

                #### 3D retrieval
                with dpg.group(horizontal=True):
                    dpg.add_text("3D retrieval option: ")

                    def callback_retrieval(sender, app_data):
                        self._3d_edit = True
                        self.edit_retrieve()
                        res_mask, _ = self.pred_res_mask(self.res_prompt)
                        res_mask = res_mask.to(self.device)
                        self.group_points(res_mask=res_mask)
                        self.renderer.gaussians._xyz.requires_grad = False

                    dpg.add_button(
                        label="retrieval", tag="_button_retrieve", callback=callback_retrieval
                    )
                    dpg.bind_item_theme("_button_retrieve", theme_button)

                    def callback_retrieval_reset(sender, app_data):
                        self.retrieval = False
                        self.need_update = True

                    dpg.add_button(
                        label="reset", tag="_button_retrieve_reset", callback=callback_retrieval_reset
                    )
                    dpg.bind_item_theme("_button_retrieve_reset", theme_button)

                    def callback_retrieval_exit(sender, app_data):
                        self.renderer.gaussians._xyz.requires_grad = True
                        self.retrieval = False
                        self._3d_edit = False
                        self.nograd_mask = None
                        self.need_update = True

                    dpg.add_button(
                        label="exit", tag="_button_retrieve_exit", callback=callback_retrieval_exit
                    )
                    dpg.bind_item_theme("_button_retrieve_exit", theme_button)

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )

                # 设置pose的
                def call_set_pose(sender, app_data):
                    self.setting_pose = int(app_data)
                    self.need_update = True

                dpg.add_combo(
                    label="Pose List",
                    callback=call_set_pose,
                    tag="pose_combo",
                )
                dpg.add_text("image name", tag="_log_img_name")

        ### register camera handler
        def callback_click_handler(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            x = app_data
            print("x:", x)

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]
            # print(dx, dy)

            self.cam.orbit(dx / self.W, -dy / self.H)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        _v = -100

        def callback_set_w():
            # normal = np.array([0, 1, 0], dtype=np.float32)
            # self.cam.center += 0.0005 * (self.cam.pose @ normal)
            self.cam.pan(0, 0, _v)

        def callback_set_s():
            self.cam.pan(0, 0, -_v)

        def callback_set_a():
            self.cam.pan(_v, 0, 0)
            # self.need_update = True
            # normal = np.array([1, 0, 0], dtype=np.float32)
            # self.cam.center -= 0.0005 * (self.cam.pose @ normal)
        def callback_set_d():
            self.cam.pan(-_v, 0, 0)

        def callback_set_q():
            self.cam.pan(0, _v, 0)

        def callback_set_e():
            self.cam.pan(0, -_v, 0)

        def callback_set_t():
            print('tttt')
            # self.cam.up *= -1
            self.cam.orbit(0, 0, 180)

        @torch.no_grad()
        def obj_move(dir: np.array):
            _mv_scale = 0.1
            ret_mask = ~self.nograd_mask
            self.renderer.gaussians._xyz[ret_mask] += torch.from_numpy(dir * _mv_scale).to(self.device)

        #### object move
        def callback_set_j():
            if self._3d_edit:
                # diff = self.cam.rot.as_matrix() @ np.array([-1, 0, 0], dtype=np.float32) * 0.1
                obj_move(np.array([-1, 0, 0], dtype=np.float32))
                self.need_update = True

        def callback_set_l():
            if self._3d_edit:
                obj_move(np.array([1, 0, 0], dtype=np.float32))
                self.need_update = True

        def callback_set_i():
            if self._3d_edit:
                obj_move(np.array([0, -1, 0], dtype=np.float32))
                self.need_update = True

        def callback_set_k():
            if self._3d_edit:
                obj_move(np.array([0, 1, 0], dtype=np.float32))
                self.need_update = True

        def callback_set_u():
            if self._3d_edit:
                obj_move(np.array([0, 0, -1], dtype=np.float32))
                self.need_update = True

        def callback_set_o():
            if self._3d_edit:
                obj_move(np.array([0, 0, 1], dtype=np.float32))
                self.need_update = True

        def callback_key_press(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            funcs = {
                dpg.mvKey_W: callback_set_w,  # w
                dpg.mvKey_A: callback_set_a,  # a
                dpg.mvKey_S: callback_set_s,  # s
                dpg.mvKey_D: callback_set_d,  # d
                dpg.mvKey_Q: callback_set_q,  # q
                dpg.mvKey_E: callback_set_e,  # e
                dpg.mvKey_T: callback_set_t,  # t
                dpg.mvKey_J: callback_set_j,  # j
                dpg.mvKey_L: callback_set_l,  # l
                dpg.mvKey_I: callback_set_i,  # i
                dpg.mvKey_K: callback_set_k,  # k
                dpg.mvKey_U: callback_set_u,  # u
                dpg.mvKey_O: callback_set_o,  # o
                # 73: self.cam.move_up, # i
                # 75: self.cam.move_down, # k
            }

            if app_data in funcs.keys():
                # print(app_data)
                print(self.cam.get_cam_location())
                funcs[app_data]()
                self.need_update = True

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle,
                                       callback=callback_camera_drag_pan)
            dpg.add_key_press_handler(callback=callback_key_press)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                        callback=callback_click_handler)

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,
                                    0,
                                    0,
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,
                                    0,
                                    0,
                                    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding,
                                    0,
                                    0,
                                    category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def add_dpg_list(self, camera_list, step=10):
        selected_cameras = camera_list[::step]
        self.gui_poses = [
            get_c2w_with_RT(camera.R, camera.T) for camera in selected_cameras
        ]
        list_gui_poses = list(range(len(self.gui_poses)))
        dpg.configure_item("pose_combo", items=list_gui_poses)

    def encode_text(self, text):
        if self.model_lang is None:
            print("model_lange need to be loaded first!")
            return
        langmodel_feature = self.model_lang.forward_text(
            [text])["last_hidden_state_eot"]
        self.clip_feature, self.clip_feature_bias = self.cls_embd.text_embedding_align(
            langmodel_feature)

    # def text_clip_encode(self, text):
    #     _text_features = self.clip_pretrained.encode_text(clip.tokenize([text]).cuda()).type(torch.float32)
    #     self.clip_feature = (_text_features / _text_features.norm(dim=-1, keepdim=True)).cuda()

    def load_models(self):

        if self.enable_sd:
            if self.enable_lods:
                print(f"[INFO] loading SD_LODS...")
                from guidance.sd_inpainting_lods_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD_LODS!")

            elif self.enable_sdxl:
                print(f"[INFO] loading SDXL...")
                from guidance.sdxl_utils import StableDiffusionXL
                self.guidance_sd = StableDiffusionXL(self.device)
                print(f"[INFO] loaded SDXL!")
            else:
                print(f"[INFO] loading SD inpainting...")
                from guidance.sd_inpainting_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD inpainting!")

        # sds text enbedding prepare
        with torch.no_grad():
            self.guidance_sd.get_text_embeds(self.prompt, self.negative_prompt)
        if self.enable_lods:
            self.guidance_sd.init_embedding_optimizer()

        print(f"[INFO] loading Model Lang...")
        self.model_lang = EVA02CLIP(
            clip_model="EVA02-CLIP-bigE-14-plus",
            cache_dir=None,
            # dtype="float16",
        ).cuda()
        self.model_lang.load_state_dict(
            torch.load('./models/model_language.pth'))
        self.cls_embd = VisionLanguageAlign(256, 1024).cuda()
        self.cls_embd.load_state_dict(
            torch.load('./models/class_embed.pth'))
        print(f"[INFO] loaded Model Lang!")

        self.enable_res = True
        if self.enable_res:
            print(f"[INFO] loading RES Model...")
            from guidance.res_model import RES_MODEL
            self.guidance_res = RES_MODEL(self.device)
            print(f"[INFO] loaded RES Model!")

        if self.opt.target_prompt is not None:
            self.clip_prompt = self.opt.target_prompt
            self.encode_text(self.clip_prompt)

    def prepare_train(self):
        # output dir prepare
        self.save_guidance_folder = os.path.join(self.workspace, "guidance")
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.save_guidance_folder, exist_ok=True)

        self.step = 0
        # setup training
        # self.renderer.gaussians.training_setup(self.opt)
        self.renderer.gaussians.finetune_sh_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # load Camera poses
        if os.path.exists(os.path.join(self.opt.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.opt.source_path,
                                                          self.opt.images,
                                                          None)
        elif os.path.exists(
                os.path.join(self.opt.source_path, "transforms_train.json")):
            print(
                "Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                self.opt.source_path, self.opt.white_background, self.opt.eval)
        else:
            scene_info = sceneLoadTypeCallbacks["ScanNet"](
                self.opt.source_path)

        args = argparse.Namespace()
        args.resolution = 8
        args.data_device = self.device
        self.all_cameras = cameraList_from_camInfos(scene_info.train_cameras,
                                                    1, args)

        # scene_info.train_cameras 是相机的list
        if self.gui:
            self.add_dpg_list(self.all_cameras, step=self.camera_select_step)

        if not self.gui:
            self.pre_compute_relative_cameras()

    def pred_res_mask(self, prompt, width=512, height=512):
        with torch.no_grad():
            out = self.render_once(width=width, height=height)
            out_image = out["image"]  #[3,H,W]

            res_mask, pred_image = self.guidance_res.predict_res_mask(
                out_image.cpu(), prompt)
            mask_out_expand = res_mask.expand_as(pred_image)
            viz_images = torch.cat(
                [pred_image.unsqueeze(0),
                 mask_out_expand.unsqueeze(0)], dim=0)
            # save_image(viz_images, "testfolder/"+ prompt + ".png")
        return res_mask, pred_image

    def group_points(self, res_mask):
        # res_mask [512,512,1]

        target_mask = ~self.nograd_mask
        # select points base on self.nograd_mask
        relative_points = self.renderer.gaussians.get_xyz[target_mask]
        relative_points = relative_points.detach().cpu().numpy()
        # print("points_to_save.shape", relative_points.shape)
        # np.savetxt("point_cloud.txt", relative_points)

        # 应用 DBSCAN 聚类
        # eps 是邻域大小，min_samples 是形成簇所需的最小点数
        dbscan = DBSCAN(eps=0.35, min_samples=600)
        clusters = dbscan.fit_predict(
            relative_points)  # 聚类结果 【N】 (-1,0,1,2...)
        set_cluster = set(clusters)
        clusters = torch.from_numpy(clusters).to(target_mask.device)
        # 找到 target_mask 中被选中的索引
        selected_indices = torch.where(target_mask == 1)[0]

        # out = self.renderer.render(cur_cam, self.gaussain_scale_factor)
        # out_image = out["image"] #[3,H,W]
        # # save out_image and save mask

        # 更新原始 mask B
        target_mask[:] = 0  # 先将所有值设为 0

        # 根据聚类出的数目循环 渲染多帧,如果iou超过80%就保留这个cluster，然后设置为最终的mask
        for i in set_cluster:
            if i == -1:
                # -1 是噪声点
                continue

            tem_target_mask = torch.zeros_like(target_mask)
            # 应该先设置mask，然后渲染
            # 找到聚类结果中类别为 i 的索引
            cluster_indices = torch.where(clusters == i)[0]
            tem_target_mask[selected_indices[cluster_indices]] = 1
            tem_target_mask = tem_target_mask.bool()
            # self.renderer.gaussians.set_semantic_masks(tem_target_mask)
            # self.need_update = True
            # self.test_step()
            # dpg.render_dearpygui_frame()

            with torch.no_grad():
                out = self.render_once(
                    width=self.render_resolution,
                    height=self.render_resolution,
                    gaussain_scale_factor=self.gaussain_scale_factor)
                out_image = out["image"]  #[3,H,W]
                out_semantic = out['semantics'].permute(1, 2,
                                                        0).detach().reshape(
                                                            -1, SEM_DIM)

                cos_sim = self.compute_clip_cos_sim(out_semantic)

                if cos_sim.sum() == 0:
                    # 当前类在当前视角即没有任何可见语义
                    continue
                semantic_mask = cos_sim > 0
                semantic_mask = semantic_mask.to(cos_sim.device)
                semantic_mask = semantic_mask.reshape(
                    -1, self.render_resolution,
                    self.render_resolution)  # [1,512,512]

            # res_mask_expand = res_mask.expand_as(out_image)
            # semantic_mask_expand = semantic_mask.expand_as(out_image)
            # viz_images = torch.cat([out_image.unsqueeze(0) ,res_mask_expand.unsqueeze(0), semantic_mask_expand.unsqueeze(0)],dim=0)
            # save_image(viz_images, "masktest.png")

            # 计算 iou
            # if semantic_mask.sum() == 0:
            #     # 当前类在当前视角即没有任何可见语义
            #     continue
            propotion = compute_mask_ratio(semantic_mask, res_mask)
            if propotion > 0.7:
                target_mask = target_mask | tem_target_mask

        self.nograd_mask = ~target_mask
        # self.renderer.gaussians.set_semantic_masks(target_mask)

        self.need_update = True

    def visual_latent(self):
        out = self.render_once(width=self.W, height=self.H)
        semantic_latent = out['semantics'].detach().cuda()
        torch.save(semantic_latent, "testfolder/test_latent.pt")

        # 可视化
        rgb_image = semantic_latent[0:3, :, :]
        rgb_image = torch.rand_like(rgb_image)
        # rgb_image = rgb_image + 1
        # rgb_image = rgb_image/2

        # 转换Tensor以符合matplotlib的期望格式[H, W, C]
        rgb_image = rgb_image.permute(1, 2, 0).to(torch.float)

        # 确保数据在0到1的范围内 这一步不能做因为这个latent 的范围不是0~1 是-3~3左右
        # rgb_image = torch.clamp(rgb_image, 0, 1)

        rgb_image_255 = (rgb_image * 255).type(torch.uint8).cpu()
        image = Image.fromarray(rgb_image_255.numpy())

        # 保存图像


        # pca to 3
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        q = semantic_latent.permute(1, 2, 0).clone().reshape(-1,
                                                             10).cpu().numpy()
        q = pca.fit_transform(q)
        q = q.reshape(self.H, self.W, 3)


    def finetune_prompt_with_res(self, res_mask):
        print("finetuen clip prompt embedding with res")

        self.res_finetuned = True
        if self.resMLP == None:
            self.resMLP = LinearSVM(set_bias=self.clip_feature_thresh).to(
                self.device)
            self.resMLP.weight_set(self.clip_feature)

        with torch.no_grad():

            out = self.render_once(width=self.W, height=self.H)
            out_image = out["image"]  #[3,H,W]
            semantics = out['semantics']  #[10,H,W]

        semantics = semantics.permute(1, 2, 0).detach().reshape(-1, SEM_DIM)
        # 过一下self.MLP
        semantic_feas = self.renderer.MLP(semantics)[:, :300]  #[HW,256]
        if self.renderer.LUT is not None:
            semantic_feas = torch.softmax(semantic_feas * 10,
                                          dim=-1).argmax(dim=-1)
            semantic_feas = self.renderer.LUT[semantic_feas]
        semantic_feas = semantic_feas / semantic_feas.norm(dim=-1,
                                                           keepdim=True)
        semantic_feas = semantic_feas.detach()

        finetune_time = time.time()
        # finetune过程
        gt = res_mask.reshape(-1, 1)
        epoch = 0
        # while(not tracker.has_converged()):
        #     loss = self.resMLP.step(semantic_feas, gt)
        #     tracker.add_loss(loss)
        #     if epoch %500 == 0:
        #         print("epoch:",epoch,"loss:",loss)
        #     epoch += 1
        max_epoch = 8000
        target_iou_thresh = 0.9

        iou = 0
        while (epoch < max_epoch and iou < target_iou_thresh):
            loss, iou = self.resMLP.step(semantic_feas, gt)
            # tracker.add_loss(loss)
            if epoch % 500 == 0:
                print("epoch:", epoch, "loss:", loss)
            epoch += 1

        torch.cuda.synchronize()
        finetune_time = time.time() - finetune_time
        print("finetune done, loss:", loss, ",epoch:", epoch, ",iou:", iou,
              ", time:", finetune_time)

    def render(self):

        self.prepare_train()
        self.cam = OrbitCamera(self.opt.W,
                               self.opt.H,
                               r=self.opt.radius,
                               fovy=self.opt.fovy)
        # data_poses, _, _ = OrbitCamera.rand_poses()

        max_epochs = 100
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_epoch()

            if self.setting_pose != -1:
                dpg.set_value(
                    "_log_img_name",
                    f"{self.all_cameras[self.setting_pose*self.camera_select_step].image_name}"
                )
                self.cam.import_pose(self.gui_poses[self.setting_pose])
                self.need_update = True
                self.setting_pose = -1

            if self.epoch % self.opt.max_epochs == 0:
                self.training = False

            self.test_step()
            dpg.render_dearpygui_frame()

    def train(self, epochs=5):
        if epochs > 0:
            self.prepare_train()
            for i in tqdm.trange(epochs):
                self.train_epoch()
                if i % 2 == 0:
                    self.save_model(mode='model', savename=(f'epoch_{i}_'))
        self.save_model(mode='model')


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters",
                        type=int,
                        default=100,
                        help="number of iterations to train")
    parser.add_argument("--config",
                        required=True,
                        help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config),
                          OmegaConf.from_cli(extras))

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters)
