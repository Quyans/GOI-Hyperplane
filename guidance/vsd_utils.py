from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers.models.embeddings import TimestepEmbedding

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    # StableDiffusionPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler
)
from torch.cuda.amp import custom_bwd, custom_fwd


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


class VSDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="2.1",
        condition_dim=1, #pose默认是16
        hf_key=None,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype, local_files_only=True
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        

        # vsd
        self.unet_lora = copy.deepcopy(pipe.unet)
        
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)
        
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype
        )
        self.scheduler_lora = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.dtype,local_files_only=True)


        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}
        
        self.unet_lora.class_embedding = TimestepEmbedding(condition_dim, 1280)
        
        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors)
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()
        
        self.to(self.device)
        

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # directional embeddings
        # for d in ['front', 'side', 'back']:
        #     embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
        #     self.embeddings[d] = embeds
    
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def compute_grad_vsd(
        self,
        latents,
        text_embeddings,
        condition,
        guidance_scale=7.5,
    ):
        B = latents.shape[0]

        with torch.no_grad():
            # random timestamp
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            
            noise_pred_pretrain = self.unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=None,
                ).sample
            
            (
                noise_pred_pretrain_text,
                noise_pred_pretrain_uncond
            ) = noise_pred_pretrain.chunk(2)

            # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
            noise_pred_pretrain = noise_pred_pretrain_uncond + guidance_scale * (
                noise_pred_pretrain_text - noise_pred_pretrain_uncond
            )

            # use view-independent text embeddings in LoRA
            # noise_pred_est = self.unet_lora(
            #     latent_model_input,
            #     torch.cat([t] * 2),
            #     encoder_hidden_states=text_embeddings,
            #     class_labels=torch.cat(
            #         [
            #             camera_condition.view(B, -1),
            #             torch.zeros_like(camera_condition.view(B, -1)),
            #         ],
            #         dim=0,
            #     ),
            #     cross_attention_kwargs={"scale": 1.0},
            # ).sample

            # (
            #     noise_pred_est_text,
            #     noise_pred_est_uncond,
            # ) = noise_pred_est.chunk(2)

            # # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
            # noise_pred_est = noise_pred_est_uncond + 1 * (noise_pred_est_text - noise_pred_est_uncond)
                
            if True:
                noise_pred_est = self.unet_lora(
                    latents_noisy,
                    t,
                    encoder_hidden_states=text_embeddings.chunk(2)[0],
                    class_labels=condition.view(B, -1),
                    cross_attention_kwargs={"scale": 1.0},
                ).sample
                
                if self.scheduler_lora.config.prediction_type == "v_prediction":
                    alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                        device=latents_noisy.device, dtype=latents_noisy.dtype
                    )
                    alpha_t = alphas_cumprod[t] ** 0.5
                    sigma_t = (1 - alphas_cumprod[t]) ** 0.5

                    noise_pred_est = latents_noisy * sigma_t.view(
                        -1, 1, 1, 1
                    ) + noise_pred_est * alpha_t.view(-1, 1, 1, 1)
                
            noise_org = noise_pred_est
            
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        
        grad = w * (noise_pred_pretrain - noise_org)
        grad = torch.nan_to_num(grad)
        return grad

    
    def train_lora(
        self,
        latents,
        text_embeddings,
        condition,
    ):
        B = latents.shape[0]
        latents = latents.detach()

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        # assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        
        noise_pred = self.unet_lora(
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings.chunk(2)[0],
            class_labels=condition.view(B, -1),
            cross_attention_kwargs={"scale": 1.0},
        ).sample
        return F.mse_loss(noise_pred, target, reduction="mean")

    
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: float,
        t: float,
        encoder_hidden_states: float,
        class_labels = None,
        cross_attention_kwargs = None,
    ) -> float:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)
    


    def train_step(
        self,
        pred_rgb,
        vsd_condition, #pose
        step_ratio=None,
        guidance_scale=7.5,
        as_latent=False,
        save_guidance_path=None
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
        grad = self.compute_grad_vsd(
            latents, embeddings, vsd_condition, guidance_scale=guidance_scale
        )
        
        loss_vsd = SpecifyGradient.apply(latents, grad)

        loss_lora = self.train_lora(latents, embeddings, vsd_condition)

        return loss_vsd.mean() * guidance_scale + loss_lora.mean() * 1


    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    1,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings
            ).sample

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = VSDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
