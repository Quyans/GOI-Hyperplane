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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, SemanticModel
from torch.nn.functional import softmax, cosine_similarity, log_softmax
import numpy as np
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def kmeans(x, ncluster, niter=10):
    '''
    x : torch.tensor(data_num,data_dim)
    ncluster : The number of clustering for data_num
    niter : Number of iterations for kmeans
    '''
    N, D = x.size()
    x /= x.norm(dim=1, keepdim=True)  # normalize each data point
    centers = x[torch.randperm(N)[:ncluster]]  # init clusters at random
    for _ in range(niter):
        centers /= centers.norm(dim=1, keepdim=True)
        distances = x @ centers.T
        assignments = distances.argmax(1)
        # move each codebook element to be the mean of the pixels that assigned to it
        centers = torch.stack([x[assignments == k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(centers), dim=1)
        ndead = nanix.sum().item()
        # print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
        centers[nanix] = x[torch.randperm(N)[:ndead]]  # re-init dead clusters
    return centers


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # torch.autograd.set_detect_anomaly = True
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sem_dim)
    semantic_MLP = SemanticModel(dim_in=dataset.sem_dim, dim_out=dataset.tab_len, num_layer=1, use_bias=True)
    sem_opt = torch.optim.Adam(semantic_MLP.parameters(), lr=0.003)
    lut = torch.nn.Parameter(torch.rand((dataset.tab_len, dataset.ape_dim), device="cuda", requires_grad=True) * 0.03)
    lut_opt = torch.optim.Adam([lut], lr=0.001)

    scene = Scene(dataset, gaussians, 1)
    gaussians.finetune_sh_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # count kmeans time
    iter_start.record()
    tot = torch.cat(
        [kmeans(x.semantic['ape'].permute(1, 2, 0).reshape(-1, dataset.ape_dim).unique(dim=0).cuda(), 80) for x in
         scene.getTrainCameras()[::8]], 0)
    tot_k = kmeans(tot, dataset.tab_len)
    lut.data = tot_k.float().clone().detach().requires_grad_(True)
    del tot
    iter_end.record()
    print(f"Kmeans time: {iter_start.elapsed_time(iter_end) / 1000:.2f}s")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, sem_feature, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
            "semantics"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        # gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        # ssim_loss = ssim(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)

        sem_feature = sem_feature.permute(1, 2, 0).reshape(-1, dataset.sem_dim)
        sem_label = semantic_MLP(sem_feature)
        sem_label = softmax(sem_label, dim=-1)  ###
        gtl = viewpoint_cam.semantic['ape'].to("cuda").float()

        gtl = gtl.permute(1, 2, 0).reshape(-1, dataset.ape_dim)
        gtl /= gtl.norm(dim=1, keepdim=True)
        lut1 = lut / lut.norm(dim=1, keepdim=True)
        sim = gtl @ lut1.T

        sim_val = sim.max(dim=1, keepdim=True)[0]
        label = (sim == sim_val).float().detach()
        lab = torch.nn.MSELoss()(sem_label, label) * 50
        sl = (1 - sim_val.mean())
        recc = 1 - cosine_similarity(lut[sem_label.argmax(-1)], gtl, dim=-1).mean()
        t = 1 if iteration < 1000 else 2
        anneal = sim * t
        b = softmax(anneal, dim=1) * log_softmax(anneal, dim=1)
        sl1 = -1.0 * b.sum(dim=-1).mean()
        # index = gtl.sum(-1).abs() > 0.002
        # sem_loss = torch.nn.MSELoss()(sem_label[index], label[index]) + 1 - cosine_similarity(lut[sem_label.argmax(-1)][index], gtl[index], dim=-1).mean()
        sem_loss = lab + sl + 0.3 * sl1 + recc
        if iteration % 100 == 1:
            print(
                f"iter {iteration}, sem_loss: {sem_loss:.6f}. {lab:.6f}, {sl:.6f},{sl1:.4f} {recc:.5f},{sim_val.min()} {torch.unique(label.argmax(dim=1)).shape[0]}")
        loss = sem_loss
        loss.backward(retain_graph=True)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, 0, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                semantic_MLP.save(
                    os.path.join(scene.model_path, f'point_cloud/iteration_{iteration}', "semantic_MLP.pt"))
                torch.save(lut, os.path.join(scene.model_path, f'point_cloud/iteration_{iteration}', "LUT.pt"))

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                sem_opt.step()
                sem_opt.zero_grad(set_to_none=True)
                lut_opt.step()
                lut_opt.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=12652)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 1500])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 1500])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
