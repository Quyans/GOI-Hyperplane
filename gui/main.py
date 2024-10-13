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
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, interpolate_transform_matrices
from utils.image_utils import apply_mask, compute_mask_ratio, calculate_iou, calculate_mean_pixel_accuracy, calculate_mean_precision, generate_video, clip_color
import argparse
from PIL import Image

from sklearn.cluster import DBSCAN
from torchvision.utils import save_image
import glob


from ext import EVA02CLIP, VisionLanguageAlign
from networks import LinearSVM, ConvergenceTracker
import cv2

SEM_DIM = 10


def resize_image(img, Height, Width):
    if img.shape[0] != Height or img.shape[1] != Width:
        return cv2.resize(img, (Width, Height), interpolation=cv2.INTER_AREA)
    return img


class ClipSimMeasure:
    def __init__(self):
        self.clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)
        self.canon = ["object", "things", "stuff", "texture"]
        self.feature_dim = 512
        self.device = torch.device("cuda")
        self.loaded = False
    
    def load_model(self):
        # no need delayed loading
        self.loaded = True
        return

    def encode_text(self, text):
        text = clip.tokenize([text] + self.canon).to(self.device)
        with torch.no_grad():
            text_features = self.clip_pretrained.encode_text(text).type(torch.float32)
            text_features = (text_features / text_features.norm(dim=-1, keepdim=True)).to(self.device)
        self.text_feature = text_features
        # return text_features
        
    def compute_similarity(self, semantic_feature):
        logit = semantic_feature @ self.text_feature.T
        positive_vals = logit[..., 0:1]  # rays x 1
        negative_vals = logit[..., 1:]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.canon))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2, should be argmin
        cos_sim = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.canon), 2))[:, 0, 0]
        return cos_sim


class ApeSimMeasure:
    def __init__(self):
        self.model_lang = None
        self.cls_embd = None
        self.feature_dim = 256
        self.device = torch.device("cuda")
        self.loaded = False

    def load_model(self):
        if self.loaded:
            return
        self.model_lang = EVA02CLIP(
            clip_model="EVA02-CLIP-bigE-14-plus",
            cache_dir=None,
            # dtype="float16",
        ).to(self.device)
        self.model_lang.load_state_dict(torch.load('./models/model_language.pth'))
        self.cls_embd = VisionLanguageAlign(256, 1024).to(self.device)
        self.cls_embd.load_state_dict(torch.load('./models/class_embed.pth'))
        self.loaded = True

    def encode_text(self, text):
        if self.model_lang is None:
            print("The VLM needs to be loaded first!")
            return
        text_feature = self.model_lang.forward_text([text])["last_hidden_state_eot"]
        text_aligned_feature, _ = self.cls_embd.text_embedding_align(text_feature)
        self.text_feature = text_aligned_feature
        
    def compute_similarity(self, semantic_feature):
        logit, manual_bias = self.cls_embd.compute_dot_product_logit_betweenTandI_manualbias(
            semantic_feature.to(self.device), self.text_feature.to(self.device))
        cos_sim = (logit).sigmoid().squeeze(-1)
        return cos_sim

class GUI:

    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui  # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        if opt.vlm.lower() == 'clip':
            self.vlm = ClipSimMeasure()
        elif opt.vlm.lower() == 'ape':
            self.vlm = ApeSimMeasure()
        else:
            NotImplementedError, f"{opt.vlm} is not supported"

        self.render_resolution = 512

        self.render_h = 0
        self.render_w = 0

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.buffer_sem = np.zeros((self.W, self.H, SEM_DIM), dtype=np.float32)
        self.sim_coloring = True
        self.sim_binary = False
        
        self.need_update = True  # update buffer_image
        self.loading = False
        
        self.white_background = opt.white_background

        self.setting_pose = -1

        # workspace
        self.workspace = os.path.join(self.opt.outdir, self.opt.save_path)

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.enable_res = True
        self.enable_sd = True
        self.enable_sdxl = False
        self.enable_lods = False
        self.enable_zero123 = False
        self.enable_rgb = True

        self.strength = 0.99

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree, white_background=self.white_background,)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.color_overlay_ratio = 0.5
        self.clip_feature_thresh = 0.86

        # input text
        self.prompt = ""
        self.negative_prompt = ""
        self.clip_prompt = ''
        self.eval_scene_load = ""
        self.scene_load = ""
        self.save_view_name = "current_view"

        self.res_prompt = ""

        # 3d manipulation
        self.gs_index = None
        self.retrieved = False
        self.motion = None

        # training stuff
        self.training = False
        self.evaluating = False
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
        self.anchor_pose_list = []
        self.render_pose_list = []
        self.render_pose_path = "render_path"

        self.gt_masks = None

        self.camera_select_step = 1

        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints":
            [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        self.scene_load = self.opt.load
        self.eval_scene_load = self.opt.source_path

        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt
        if self.opt.target_prompt is not None:
            self.clip_prompt = self.opt.target_prompt
            self.vlm.encode_text(self.clip_prompt)
        if self.opt.target_res_prompt is not None:
            self.res_prompt = self.opt.target_res_prompt

        # override if provide a checkpoint, or initialize from scratch
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)
        else:
            self.renderer.initialize(num_pts=self.opt.num_pts)

        self.resMLP = None
        self.res_finetuned = False

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def load_new_eval_data(self, scene_path):
        opt.source_path = self.eval_scene_load
        self.opt = opt
        self.prepare_train()

    def load_new_scene(self, scene_path):
        # 保证scene_path是一个文件夹
        # 并且保证语言模型不用重新load
        # 修改opt

        opt.load = self.scene_load
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.

        self.mode = "image"
        self.seed = "random"

        self.need_update = True  # update buffer_image
        self.setting_pose = -1

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree, white_background=self.white_background)
        self.gaussain_scale_factor = 1

        # input text
        self.prompt = ""
        self.negative_prompt = ""
        self.clip_prompt = ""
        self.res_prompt = ""

        # 3d manipulation
        self.gs_index = None
        self.retrieved = False
        self.motion = None

        # training stuff
        self.training = False
        self.evaluating = False
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

        self.gt_masks = None

        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints":
            [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt
        if self.opt.target_prompt is not None:
            self.clip_prompt = self.opt.target_prompt
            self.vlm.encode_text(self.clip_prompt)
        if self.opt.target_res_prompt is not None:
            self.res_prompt = self.opt.target_res_prompt


        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        self.resMLP = None
        self.res_finetuned = False
        old_pose = self.cam.pose
        self.prepare_train()
        self.cam.set_pose(old_pose)

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

    @torch.no_grad()
    def compute_similarity(self, embedding_feature, out_bg_mask=None):
        dec_feature = self.renderer.MLP(embedding_feature)
        if self.renderer.LUT is not None:
            sem_logit = torch.softmax(dec_feature * 10, dim=-1).argmax(dim=-1)
            sem_feature = self.renderer.LUT[sem_logit]
        else:
            sem_feature = dec_feature
        normed_feature = sem_feature / sem_feature.norm(dim=-1, keepdim=True)
        if self.res_finetuned:
            # 使用res finetune的MLP作为分割效果
            logit = self.resMLP(normed_feature.cuda()).squeeze()
            # print(logit.max(), logit.min())
            sim = (logit).sigmoid().squeeze(-1)
            thresh = 0.5
        else:
            sim = self.vlm.compute_similarity(normed_feature)
            thresh = self.clip_feature_thresh
        _bg_mask = sim < thresh
        if out_bg_mask is not None:
            out_bg_mask[:] = _bg_mask
        sim[_bg_mask] = 0
        return sim

    @torch.no_grad()
    def set_clip_mask(self):
        bg_mask = torch.zeros(self.H * self.W, dtype=torch.bool, device=self.device)
        cos_sim = self.compute_similarity(self.buffer_sem, bg_mask)
        if not self.sim_binary:
            colored_img, alpha = clip_color(cos_sim, bg_mask, height=self.H, width=self.W, thresh=0.7, res_finetuned=self.res_finetuned, coloring=self.sim_coloring, device=self.device)
            opa = alpha * self.color_overlay_ratio
            self.buffer_image = (colored_img * opa + self.buffer_image * (1 - opa)).clip(0, 1)
        else:
            mask = cos_sim > 0
            binary_mask = mask.reshape(self.H, self.W).float().unsqueeze(-1).repeat(1, 1, 3)
            self.buffer_image = binary_mask.contiguous().clamp(0, 1).contiguous().detach().cpu().numpy()

    def compute_relative_gs_index(self):
        gs_embedding = self.renderer.gaussians.get_semantics
        gs_semantics = self.compute_similarity(gs_embedding)
        rel_gs_index = gs_semantics > 0  #[N]
        rel_gs_index = rel_gs_index.to(gs_semantics.device)
        return rel_gs_index

    @torch.no_grad()
    def pre_compute_relative_cameras(self):
        print("precompute...")
        self.renderer.gaussians.set_semantic_masks()
        self.need_update = True
        # 将相关的3d GS球mask出来并且设置 (1-mask) 的GS require_grad=False

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
            cos_sim = self.compute_similarity(out_semantic)

            if cos_sim.any():
                relative_pixel_number = torch.count_nonzero(cos_sim)
                max_relative_number = max(max_relative_number,
                                          relative_pixel_number)
                camera.relative_pixel_number = relative_pixel_number

                # semantic = out["semantics"].detach().permute(1, 2, 0).reshape(-1, SEM_DIM) #[HW, 10]
                # semantic_cropped = self.compute_similarity(semantic)
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
            nograd_mask_expand = apply_mask(gs._features_rest.grad, nograd_mask)
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
        croped_semantic = self.compute_similarity(gs_semantics)

        crop_mask = croped_semantic > 0
        crop_mask = crop_mask.to(croped_semantic.device)
        self.renderer.gaussians.prune_points(crop_mask)
        self.need_update = True

    @torch.no_grad()
    def edit_retrieve(self):
        self.rel_gs_index = self.compute_relative_gs_index()
        self.retrieved = True
        # self.need_update = True

    def load_gt_masks(self, cameras):
        gt_masks_dir = os.path.join(self.opt.source_path, "masks")
        gt_masks = []
        for camera in cameras:
            mask_file = os.path.join(gt_masks_dir, camera.image_name + ".png")

            mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
            mask = resize_image(mask, self.render_resolution_h,
                                self.render_resolution_w)

            mask = mask.astype(np.float32) / 255  # [H, W, 3/4]
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            gt_masks.append(mask)

        gt_masks = torch.from_numpy(np.stack(gt_masks, axis=0)).to(self.device)
        return gt_masks

    @torch.no_grad()
    def test_step(self):

        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            out = self.render_once(width=self.W, height=self.H)

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

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )
            self.buffer_sem = out['semantics'].permute(1, 2, 0).detach().cuda().reshape(-1, SEM_DIM)

            self.need_update = False
            if self.clip_prompt != '' and self.vlm.loaded == True:
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

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024, savename=None):

        if mode == 'geo':
            path = os.path.join(self.workspace, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.workspace, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3),
                                 device=self.device,
                                 dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
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
                path = os.path.join(self.workspace, savename + 'point_cloud.ply')
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
                print("self.workspace: ", self.workspace)

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

                    def callback_set_color_overlay_ratio(sender, app_data):
                        self.color_overlay_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.2f",
                        default_value=self.color_overlay_ratio,
                        callback=callback_set_color_overlay_ratio,
                    )

                with dpg.group(horizontal=True):

                    def callback_load_new_scene(sender, app_data, user_data):
                        self.loading = True
                        self.load_new_scene(self.scene_load)
                        self.loading = False

                    dpg.add_button(
                        label="load_new_scene",
                        tag="_button_load_new_scene",
                        callback=callback_load_new_scene,
                        user_data='load_new_scene',
                    )
                    dpg.bind_item_theme("_button_load_new_scene", theme_button)
                    dpg.add_input_text(
                        # label="eval_scene_load",
                        default_value=self.scene_load,
                        callback=callback_setattr,
                        user_data="scene_load",
                    )

                with dpg.group(horizontal=True):

                    def callback_load_new_eval_data(sender, app_data,
                                                    user_data):
                        self.load_new_eval_data(self.eval_scene_load)

                    dpg.add_button(
                        label="load_new_eval_data",
                        tag="_button_load_new_eval_data",
                        callback=callback_load_new_eval_data,
                        user_data='load_new_eval_data',
                    )
                    dpg.bind_item_theme("_button_load_new_eval_data",
                                        theme_button)
                    dpg.add_input_text(
                        # label="eval_scene_load",
                        default_value=self.eval_scene_load,
                        callback=callback_setattr,
                        user_data="eval_scene_load",
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
            with dpg.collapsing_header(label="Query", default_open=True):
                # lr and train button

                with dpg.group(horizontal=True):

                    dpg.add_button(label="init",
                                   tag="_button_init",
                                   callback=self.load_models_in_thread)
                    dpg.bind_item_theme("_button_init", theme_button)


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

                ## query
                with dpg.group(horizontal=True):
                    def callback_update_clip_mask(sender, app_data):
                        self.clip_prompt = app_data
                        self.vlm.encode_text(self.clip_prompt)
                        self.need_update = True

                    dpg.add_input_text(
                        label="CLIP",
                        tag="_clip_prompt",
                        default_value=self.clip_prompt,
                        on_enter=True,
                        callback=callback_update_clip_mask,
                    )

                    def callback_update_sim_color(sender, app_data):
                        self.sim_coloring = app_data
                        self.need_update = True

                    dpg.add_checkbox(
                        label="coloring",
                        default_value=self.sim_coloring,
                        callback=callback_update_sim_color,
                    )

                with dpg.group(horizontal=True):
                    def callback_update_res_prompt(sender, app_data):
                        self.res_prompt = app_data

                    dpg.add_input_text(
                        label="RES",
                        tag="_res_prompt",
                        default_value=self.res_prompt,
                        # on_enter=True,
                        callback=callback_update_res_prompt,
                    )

                    def callback_update_sim_binary(sender, app_data):
                        self.sim_binary = app_data
                        self.need_update = True

                    dpg.add_checkbox(
                        label="binary",
                        default_value=self.sim_binary,
                        callback=callback_update_sim_binary,
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
                ) ##

                with dpg.group(horizontal=True):

                    def callback_res_locate(sender, app_data):
                        res_mask, _ = self.pred_res_mask(self.res_prompt, self.W, self.H)
                        res_mask = res_mask.to(self.device)
                        self.finetune_prompt_with_res(res_mask=res_mask)
                        self.need_update = True

                    def callback_refresh_res(sender, app_data):
                        self.resMLP = None
                        self.res_finetuned = False
                        self.need_update = True

                    dpg.add_text("RES:")
                    dpg.add_button(label="res_loc", callback=callback_res_locate)

                    dpg.add_button(label="refresh", callback=callback_refresh_res)

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
                    def callback_start_evalate(sender, app_data):
                        if self.evaluating:
                            self.evaluating = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.evaluating = True
                            dpg.configure_item("_button_train", label="stop")

                    dpg.add_text("Evaluate:")
                    dpg.add_button(label="start",
                                   tag="_button_eval",
                                   callback=callback_start_evalate)
                    dpg.bind_item_theme("_button_eval", theme_button)

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

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    dpg.add_button(label="start",
                                   tag="_button_train",
                                   callback=callback_train)
                    dpg.bind_item_theme("_button_train", theme_button)
                # Temporal not using direct deletion
                # with dpg.group(horizontal=True):
                #     dpg.add_text("Edit option: ")

                #     def callback_delete(sender, app_data):
                #         self.edit_delete()

                #     dpg.add_button(label="delete",
                #                    tag="_button_delete",
                #                    callback=callback_delete)
                #     dpg.bind_item_theme("_button_delete", theme_button)

                #### 3D retrieval
                # 1. ret, can move, 2. seg & del 3. reset (reset to origin) 4. exit
                self.use_cluster = False
                with dpg.group(horizontal=True):
                    dpg.add_text("3D manipulation option: ")

                    def callback_retrieval(sender, app_data):
                        self.edit_retrieve()
                        if self.use_cluster:
                            res_mask, _ = self.pred_res_mask(self.res_prompt)
                            res_mask = res_mask.to(self.device)
                            self.group_points(res_mask=res_mask)
                        self.renderer.gaussians._xyz.requires_grad = False
                        self.motion = torch.zeros_like(self.renderer.gaussians._xyz)
                        print('[INFO] Object retrieved.')

                    dpg.add_button(
                        label="retrieve", tag="_button_retrieve", callback=callback_retrieval
                    )
                    dpg.bind_item_theme("_button_retrieve", theme_button)

                    def callback_3dseg(sender, app_data):
                        self.gs_index = self.rel_gs_index
                        self.need_update = True

                    dpg.add_button(
                        label="seg", tag="_button_3d_seg", callback=callback_3dseg
                    )
                    dpg.bind_item_theme("_button_3d_seg", theme_button)

                    def callback_3ddel(sender, app_data):
                        self.gs_index = ~self.rel_gs_index
                        self.need_update = True

                    dpg.add_button(
                        label="del", tag="_button_3d_del", callback=callback_3ddel
                    )
                    dpg.bind_item_theme("_button_3d_del", theme_button)

                    def callback_retrieval_exit(sender, app_data):
                        self.renderer.gaussians._xyz -= self.motion
                        self.motion = None
                        self.renderer.gaussians._xyz.requires_grad = True
                        self.retrieved = False
                        self.rel_gs_index = None
                        self.gs_index = None
                        self.need_update = True
                        self.renderer.gaussians.set_semantic_masks()
                        print('[INFO] Object retrieval exit.')

                    dpg.add_button(
                        label="exit", tag="_button_retrieve_exit", callback=callback_retrieval_exit
                    )
                    dpg.bind_item_theme("_button_retrieve_exit", theme_button)

                    def callback_retrieval_reset(sender, app_data):
                        self.renderer.gaussians._xyz -= self.motion
                        self.motion.zero_()
                        self.gs_index = None
                        self.need_update = True

                    dpg.add_text("aux: ")
                    dpg.add_button(
                        label="reset", tag="_button_retrieve_reset", callback=callback_retrieval_reset
                    )
                    dpg.bind_item_theme("_button_retrieve_reset", theme_button)

                    def callback_use_cluster(sender, app_data):
                        self.use_cluster = app_data
                        print(self.use_cluster, 777)

                    dpg.add_checkbox(
                        label="use_cluster",
                        default_value=self.use_cluster,
                        callback=callback_use_cluster,
                    )

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

                def call_add_render_pose(sender, app_data):
                    ahchor_pose = self.cam.pose
                    self.anchor_pose_list.append(ahchor_pose)
                    print("add render pose:", len(self.anchor_pose_list))

                def call_inter_poses(sender, app_data):
                    self.render_pose_list = interpolate_transform_matrices(
                        self.anchor_pose_list, 10)
                    # 保存pose_list

                    save_path = os.path.join(self.workspace,
                                             self.render_pose_path + ".npz")
                    # 使用savez保存数组列表到.npz文件
                    np.savez(save_path, *self.render_pose_list)

                    # 从.npz文件加载数组列表
                    # loaded_data = np.load('arrays_list.npz')
                    # loaded_list_of_arrays = [loaded_data[f'arr_{i}'] for i in range(len(list_of_arrays))]
                    # 输出以验证
                    # print("原始列表中的数组:")
                    # for arr in list_of_arrays:
                    #     print(arr)
                    # print("加载后的列表中的数组:")
                    # for arr in loaded_list_of_arrays:
                    #     print(arr)

                    print("Inter pose finush, total",
                          len(self.render_pose_list), "poses")
                    # 对相机四元数插值

                def call_load_inter_poses(sender, app_data):
                    # 从.npz文件加载数组列表
                    load_path = os.path.join(self.workspace,
                                             self.render_pose_path + ".npz")

                    loaded_data = np.load(load_path)
                    self.render_pose_list = [
                        loaded_data[f'arr_{i}']
                        for i in range(len(loaded_data.files))
                    ]
                    print("load render path: ", len(self.render_pose_list))

                def call_render_interposes(sender, app_data):
                    self.save_video_folder = os.path.join(
                        self.workspace, "video")
                    self.render_video(save_path=self.save_video_folder)
                    print("render video finush")

                def call_clear_interposes(sender, app_data):
                    self.anchor_pose_list = []
                    self.render_pose_list = []
                    print("clear pose")

                dpg.add_combo(
                    label="Pose List",
                    callback=call_set_pose,
                    tag="pose_combo",
                )
                dpg.add_text("image name", tag="_log_img_name")

                def callback_update_res_prompt(sender, app_data):
                    self.render_pose_path = app_data

                dpg.add_input_text(
                    label="pose_path",
                    tag="_pose_path",
                    default_value=self.render_pose_path,
                    # on_enter=True,
                    callback=callback_update_res_prompt,
                )
                with dpg.group(horizontal=True):
                    dpg.add_button(label="add_render_pose",
                                   callback=call_add_render_pose)
                    dpg.add_button(label="inter_poses",
                                   callback=call_inter_poses)
                    dpg.add_button(label="start_render",
                                   callback=call_render_interposes)
                    dpg.add_button(label="clear_pose",
                                   callback=call_clear_interposes)
                    dpg.add_button(label="load_render_path",
                                   callback=call_load_inter_poses)

        ### register camera handler
        def callback_click_handler(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            # x = app_data
            # print("x:", x)

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

        def obj_move(x, y, z):
            if self.retrieved:
                _mov = torch.from_numpy(np.array([x, y, z], dtype=np.float32) * 0.1).to(self.device)
                ret_mask = self.rel_gs_index
                self.motion[ret_mask] += _mov
                self.renderer.gaussians._xyz[ret_mask] += _mov
        
        def callback_key_press(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return
            
            _v = -100
            funcs = {
                dpg.mvKey_W: lambda: self.cam.pan(0, 0, _v),  # w
                dpg.mvKey_A: lambda: self.cam.pan(_v, 0, 0),  # a
                dpg.mvKey_S: lambda: self.cam.pan(0, 0, -_v),  # s
                dpg.mvKey_D: lambda: self.cam.pan(-_v, 0, 0),  # d
                dpg.mvKey_Q: lambda: self.cam.pan(0, _v, 0),  # q
                dpg.mvKey_E: lambda: self.cam.pan(0, -_v, 0),  # e

                dpg.mvKey_J: lambda: obj_move(-1, 0, 0),  # j
                dpg.mvKey_L: lambda: obj_move(1, 0, 0),  # l
                dpg.mvKey_I: lambda: obj_move(0, -1, 0),  # i
                dpg.mvKey_K: lambda: obj_move(0, 1, 0),  # k
                dpg.mvKey_U: lambda: obj_move(0, 0, -1),  # u
                dpg.mvKey_O: lambda: obj_move(0, 0, 1),  # o
                
                dpg.mvKey_T: lambda: self.cam.orbit(0, 0, 180),  # t
            }

            if app_data in funcs.keys():
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

    def prepare_train(self):
        # output dir prepare
        self.save_guidance_folder = os.path.join(self.workspace, "guidance")
        os.makedirs(self.save_guidance_folder, exist_ok=True)

        self.step = 0
        # setup training
        self.renderer.gaussians.finetune_sh_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = 3
        self.optimizer = self.renderer.gaussians.optimizer

        # load Camera poses
        if os.path.exists(os.path.join(self.opt.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.opt.source_path, self.opt.images, eval=False, load_sem=False)
        elif os.path.exists(
                os.path.join(self.opt.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](self.opt.source_path, self.opt.white_background, self.opt.eval, load_sem=False)
        else:
            scene_info = sceneLoadTypeCallbacks["ScanNet"](
                self.opt.source_path)

        args = argparse.Namespace()
        args.resolution = 1  #因为目前的mask是2分之一的
        args.data_device = self.device

        self.all_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1, args)

        templet_camera = self.all_cameras[0]

        gt_fovx, gt_fovy = np.rad2deg(templet_camera.FoVx), np.rad2deg(
            templet_camera.FoVy)
        self.cam = OrbitCamera(self.opt.W,
                               self.opt.H,
                               r=0,
                               fovy=gt_fovy,
                               fovx=gt_fovx)
        if templet_camera.image_width > 800:
            self.render_resolution_h = templet_camera.image_height // 2
            self.render_resolution_w = templet_camera.image_width // 2
        else:
            self.render_resolution_h = templet_camera.image_height
            self.render_resolution_w = templet_camera.image_width

        # scene_info.train_cameras 是相机的list
        if self.gui:
            self.add_dpg_list(self.all_cameras, step=self.camera_select_step)

        if not self.gui:
            self.pre_compute_relative_cameras()

    def load_models(self):
        print(f"[INFO] loading VLM...")
        self.vlm.load_model()
        print(f"[INFO] loaded VLM!")

        if self.enable_res:
            print(f"[INFO] loading RES Model...")
            from guidance.res_model import RES_MODEL
            self.guidance_res = RES_MODEL(self.device)
            print(f"[INFO] loaded RES Model!")

        if self.opt.target_prompt is not None:
            self.clip_prompt = self.opt.target_prompt
            self.vlm.encode_text(self.clip_prompt)
        print(f"[INFO] loading models finished!")

    def load_models_in_thread(self):
        import threading
        # 创建并启动子线程
        thread = threading.Thread(target=self.load_models)
        thread.start()

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

        target_mask = self.rel_gs_index
        # select points base on self.rel_gs_index
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
            self.renderer.gaussians.set_semantic_masks(tem_target_mask)
            # self.need_update = True
            # self.test_step()
            # dpg.render_dearpygui_frame()

            with torch.no_grad():
                out = self.render_once(width=self.render_resolution,
                                       height=self.render_resolution)
                out_image = out["image"]  #[3,H,W]
                out_semantic = out['semantics'].permute(1, 2, 0).detach().reshape(-1, SEM_DIM)

                cos_sim = self.compute_similarity(out_semantic)

                if cos_sim.sum() == 0:
                    # 当前类在当前视角即没有任何可见语义
                    continue
                semantic_mask = cos_sim > 0
                semantic_mask = semantic_mask.to(cos_sim.device)
                semantic_mask = semantic_mask.reshape(-1, self.render_resolution,
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

        self.rel_gs_index = target_mask
        self.renderer.gaussians.set_semantic_masks(target_mask)

        self.need_update = True

    def finetune_prompt_with_res(self, res_mask):
        print("Fine tuning prompt feature with RES")

        self.res_finetuned = True
        if self.resMLP == None:
            self.resMLP = LinearSVM(set_bias=self.clip_feature_thresh, input_dim=self.vlm.feature_dim).to(
                self.device)
            self.resMLP.weight_set(self.vlm.text_feature)

        with torch.no_grad():

            out = self.render_once(width=self.W, height=self.H)
            out_image = out["image"]  #[3,H,W]
            rendered_embd = out['semantics']  #[10,H,W]

        rendered_embd = rendered_embd.permute(1, 2, 0).detach().reshape(-1, SEM_DIM)
        # 过一下self.MLP
        dec_feature = self.renderer.MLP(rendered_embd)[:, :1000]  #[HW,256]
        if self.renderer.LUT is not None:
            sem_logit = torch.softmax(dec_feature * 10, dim=-1).argmax(dim=-1)
            sem_feature = self.renderer.LUT[sem_logit]
        normed_feature = sem_feature / sem_feature.norm(dim=-1, keepdim=True)
        normed_feature = normed_feature.detach()

        finetune_time = time.time()
        # finetune过程
        gt = res_mask.reshape(-1, 1)
        epoch = 0
        # while(not tracker.has_converged()):
        #     loss = self.resMLP.step(normed_feature, gt)
        #     tracker.add_loss(loss)
        #     if epoch %500 == 0:
        #         print("epoch:",epoch,"loss:",loss)
        #     epoch += 1
        max_epoch = 8000
        target_iou_thresh = 0.9
        iou = 0

        with torch.no_grad():
            print("init_iou", self.resMLP.eval_forward(normed_feature, gt))
        while (epoch < max_epoch and iou < target_iou_thresh):
            # loss = torch.tensor(0).to(self.device)
            # iou = torch.tensor(0).to(self.device)
            if epoch % 500 == 0:

                logit = (self.resMLP(normed_feature.cuda())).squeeze()
                print(logit.max(), logit.min())
                logit = (logit).sigmoid().squeeze(-1)
                clip_thresh = 0.5
                logit_mask = logit > clip_thresh
                logit_mask = logit_mask.cpu().numpy()

                logit_mask = logit_mask.reshape(self.H, self.W)

                gt_mask = gt.bool().squeeze(-1).cpu().numpy().reshape(
                    self.H, self.W)
                # mask_np = logit_mask.cpu().numpy().squeeze(-1)
                # gt_mask_np = gt_masks[i].cpu().numpy().squeeze(-1)
                mask_image = Image.fromarray(logit_mask)
                gt_mask_image = Image.fromarray(gt_mask)

                # mask_image.save('testfolder/'+str(self.W==self.H)+'Finetune_mask_image-epoch'+str(epoch)+'.png')
                # gt_mask_image.save('testfolder/Finetune_gt_mask_image-epoch'+str(epoch)+'.png')

            loss, iou = self.resMLP.step(normed_feature, gt)

            if epoch % 500 == 0:
                print("epoch:", epoch, "loss:", loss.item(), "iou:", iou)
            # tracker.add_loss(loss)

            epoch += 1

        logit = (self.resMLP(normed_feature.cuda())).squeeze()
        print(logit.max(), logit.min())
        logit = (logit).sigmoid().squeeze(-1)
        clip_thresh = 0.5
        logit_mask = logit > clip_thresh
        logit_mask = logit_mask.cpu().numpy()

        logit_mask = logit_mask.reshape(self.H, self.W)

        gt_mask = gt.bool().squeeze(-1).cpu().numpy().reshape(self.H, self.W)
        # mask_np = logit_mask.cpu().numpy().squeeze(-1)
        # gt_mask_np = gt_masks[i].cpu().numpy().squeeze(-1)
        mask_image = Image.fromarray(logit_mask)
        gt_mask_image = Image.fromarray(gt_mask)

        torch.cuda.synchronize()
        finetune_time = time.time() - finetune_time
        print("finetune done, loss:", loss, ",epoch:", epoch, ",iou:", iou,
              ", time:", finetune_time)


    def render_video(self, save_path, num_step=1):
        # 从当前视角 渲染所有视角并且保存为video
        # 先渲染所有视角并且保存

        os.makedirs(save_path, exist_ok=True)

        for ind in range(len(self.render_pose_list)):
            pose = self.render_pose_list[ind]
            cur_cam = MiniCam(pose, self.render_resolution_w,
                              self.render_resolution_h, self.cam.fovy,
                              self.cam.fovx, self.cam.near, self.cam.far)

            bg_color = torch.tensor([0, 0, 0],
                                    dtype=torch.float32,
                                    device="cuda")
            out = self.renderer.render(cur_cam, bg_color=bg_color)
            out_img = out["image"]
            out_semantic = out["semantics"]

            bg_mask = torch.zeros(self.render_resolution_h * self.render_resolution_w, dtype=torch.bool, device=self.device)
            cos_sim = self.compute_similarity(out_semantic.permute(1, 2, 0).detach().cuda().reshape(-1, SEM_DIM), bg_mask)

            rgb = out_img.permute(1, 2, 0).contiguous().clamp(0, 1).contiguous().detach().cpu().numpy()

            if not self.sim_binary:
                colored_img, alpha = clip_color(cos_sim, bg_mask, self.render_resolution_h, self.render_resolution_w, 
                                   thresh=0.7, res_finetuned=self.res_finetuned, coloring=self.sim_coloring, device=self.device)
                opa = alpha * self.color_overlay_ratio
                final_img = (colored_img * opa + rgb * (1 - opa)).clip(0, 1)
            else:
                mask = cos_sim > 0
                binary_mask = mask.reshape(self.render_resolution_h, self.render_resolution_w).float().unsqueeze(-1).repeat(1, 1, 3)
                final_img = binary_mask.contiguous().clamp(0, 1).contiguous().detach().cpu().numpy()

            final_img = Image.fromarray((final_img * 255).astype('uint8'))
            final_img.save(f"{save_path}/{str(ind)}.png")

        def natural_sort_key(s):
            """
            Analyzes the key in a natural way to sort the strings that contain numbers.
            It converts the string into a list of mixed types (strings and integers)
            and returns it as a key for sorting.
            """
            import re
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)
            ]

        img_list = glob.glob(os.path.join(save_path, '*.png'))

        img_list.sort(
            key=natural_sort_key
        )  # Sort the images by name (ensure they are named in sequence)

        generate_video(img_list, save_path=save_path)


    def render(self):

        self.prepare_train()
        # self.cam = OrbitCamera(self.opt.W, self.opt.H, r=self.opt.radius, fovy=self.opt.fovy)
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

            if not self.loading:
                # diff thread for flushing the renderer and loading new scene, so need a mutex
                self.test_step()
            dpg.render_dearpygui_frame()

    def render_once(self, width, height, camera=None, gaussain_scale_factor=None):

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
            out = self.renderer.render(cur_cam, self.gaussain_scale_factor, gaussian_mask=self.gs_index)
        else:
            out = self.renderer.render(cur_cam, gaussain_scale_factor, gaussian_mask=self.gs_index)
        return out

    def train(self, epochs=5):
        if epochs > 0:
            self.prepare_train()
            for i in tqdm.trange(epochs):
                self.train_epoch()
                if i % 2 == 0:
                    self.save_model(mode='model', savename=(f'epoch_{i}_'))
        self.save_model(mode='model')
    
    def train_step(self, cameras):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        self.step += 1
        ##### train step lost here #####
        
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

    def eval_step(self, camera, gt_mask):

        C2W = get_c2w_with_RT(camera.R, camera.T)
        cur_cam = MiniCam(C2W, self.render_resolution_w,
                          self.render_resolution_h, self.cam.fovy,
                          self.cam.fovx, self.cam.near, self.cam.far)
        out = self.renderer.render(cur_cam)

        # 计算iou
        pred_mask = out['semantics'].permute(1, 2,
                                             0).detach().reshape(-1, SEM_DIM)
        pred_mask = self.compute_similarity(pred_mask)
        pred_mask = pred_mask > 0
        pred_mask = pred_mask.to(pred_mask.device)

        iou = calculate_iou(pred_mask, gt_mask)
        return iou

    @torch.no_grad()
    def eval_epoch(self):
        # 测试测试集的指标
        # miou，Mean Pixel Accuracy，Mean Precision，Mean Average Precision， cause time， FPS，
        # 先只计算Miou

        print("test epoch")

        gt_masks = self.gt_masks
        print("test epoch")

        total_iou = 0.0
        total_mpa = 0.0
        total_mp = 0.0
        for i in range(len(self.all_cameras)):
            camera = self.all_cameras[i]

            pose = get_c2w_with_RT(camera.R, camera.T)

            cur_cam = MiniCam(pose, self.render_resolution_w,
                              self.render_resolution_h, self.cam.fovy,
                              self.cam.fovx, self.cam.near, self.cam.far)

            # bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
            bg_color = torch.tensor([1, 1, 1] if self.opt.white_background else [0, 0, 0],
                                    dtype=torch.float32,
                                    device="cuda")
            out = self.renderer.render(cur_cam, bg_color=bg_color)

            out_image = out["image"]  #[3,H,W]

            pred_mask = out['semantics'].permute(1, 2, 0).detach().reshape(
                -1, SEM_DIM)
            pred_mask = self.compute_similarity(pred_mask)
            pred_mask = pred_mask > 0
            pred_mask = pred_mask.to(pred_mask.device)
            pred_mask = pred_mask.reshape(self.render_resolution_h,
                                          self.render_resolution_w,
                                          -1)  # [512,512,1]

            iou = calculate_iou(gt_masks[i], pred_mask)
            mpa = calculate_mean_pixel_accuracy(gt_masks[i], pred_mask)
            mp = calculate_mean_precision(gt_masks[i], pred_mask)

            mask_np = pred_mask.cpu().numpy().squeeze(-1)
            gt_mask_np = gt_masks[i].cpu().numpy().squeeze(-1)
            mask_image = Image.fromarray(mask_np)
            gt_mask_image = Image.fromarray(gt_mask_np)

            # mask_image.save('testfolder/mask_image'+str(i)+'.png')
            # gt_mask_image.save('testfolder/gt_mask_image'+str(i)+'.png')

            print("iou:", iou, " mpa:", mpa, " mp:", mp)
            total_iou += iou
            total_mpa += mpa
            total_mp += mp
        avg_iou = total_iou / len(self.all_cameras)
        avg_mpa = total_mpa / len(self.all_cameras)
        avg_mp = total_mp / len(self.all_cameras)
        print("############################")
        print("avg_iou:", avg_iou, " avg_mpa:", avg_mpa, " avg_mp:", avg_mp)


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters",
                        type=int,
                        default=100,
                        help="number of iterations to train")
    parser.add_argument("--config", '-c',
                        default="gui/configs/default.yaml",
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
