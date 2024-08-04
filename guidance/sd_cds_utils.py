from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import os 

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=False,
        vram_O=False,
        sd_version="2.1",
        hf_key=None,
        t_range=[0.1, 0.7],
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

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype, local_files_only=True
        )

        del pipe
        self.min_t = t_range[0]
        self.max_t = t_range[1]
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.min_t)
        self.max_step = int(self.num_train_timesteps * self.max_t)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}
        
        # CDS fixed latent Noise 
        # self.base_noise = torch.rand((4,64,64),device=self.device)
        

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds
    
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

    def predict_start_from_noise(self, latents_noisy, t, noise_pred):
        # latents_noisy是 latents 加噪之后的xt
        total_timesteps = self.max_step - self.min_step + 1
        
        index = total_timesteps - t.to(latents_noisy.device) - 1 
        b = noise_pred.shape[0]
        a_t = self.alphas[index].reshape(b,1,1,1).to(self.device) #a_t 即alpha_t
        sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas)

        sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
        latents_recon = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
        return latents_recon

    def predict_start_from_noise_cds(self, latents_noisy, t, noise_pred):
        batch_size = latents_noisy.shape[0]
        sigma = ((2*t) ** 0.5).view(batch_size, 1, 1, 1)
        latents_recon = latents_noisy - sigma * noise_pred
        return latents_recon
    
    @torch.no_grad()
    def refine(self, pred_rgb,
               guidance_scale=100, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings,
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs

    def add_noise(self, latents, noise, sigma_t):
        sigma_t = sigma_t.to(latents.device)
        
        # sigma_t = (2*t) ** 0.5
        coefficient = sigma_t
        coefficient = coefficient.flatten()
        
        while len(coefficient.shape) < len(latents.shape):
            coefficient = coefficient.unsqueeze(-1)
        
        noisy_samples = latents + coefficient * noise
        return noisy_samples

    def train_step(
        self,
        pred_rgb,
        step_ratio=None,
        guidance_scale=100,
        as_latent=False,
        vers=None, hors=None,
        save_guidance_path=None,
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

        with torch.no_grad():
            # if step_ratio is not None:
            #     # dreamtime-like
            #     # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            #     t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            #     t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            # else:
            #     t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)


            t2 =  self.max_t - (self.max_t - self.min_t)* np.sqrt(step_ratio)
            
            delta = 0.1
            Delta = 0.2
            
            t1 = np.random.uniform(low=t2+delta, high=t2+Delta)
            # # 这里t的shape转为 [batch_size] -> [1,batch_size]
            # t = t.unsqueeze(0)
            t1 = torch.from_numpy(np.array([t1]* batch_size)).to(self.dtype).to(self.device)
            t2 = torch.from_numpy(np.array([t2]* batch_size)).to(self.dtype).to(self.device)
            
            t1_step = t1 * self.num_train_timesteps
            t2_step = t2 * self.num_train_timesteps
            t1_step = t1_step.to(torch.int64)
            t2_step = t2_step.to(torch.int64)

            sigma_t1 = ((2*t1) ** 0.5).view(batch_size, 1, 1, 1)
            sigma_t2 = ((2*t2) ** 0.5).view(batch_size, 1, 1, 1)
            # sigma_t2 = self.alphas[t2_step].view(batch_size, 1, 1, 1)

            # predict the noise residual with unet, NO grad!
            # add noise
            # noise = self.base_noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            
            noise = torch.randn_like(latents)

            latents_noisy_t1 = self.add_noise(latents, noise, sigma_t1)
            # [batch,c,h,w] -> [1,batch,c,h,w]
            # latents_noisy_t1 = latents_noisy_t1.unsqueeze(0) # sd的unet输入是[2*batch,c,h,w]
            # pred noise
            latent_model_input_t1 = torch.cat([latents_noisy_t1] * 2) # sd的unet输入是[2*batch,c,h,w]
            tt1 = torch.cat([t1_step] * 2)
            tt2 = torch.cat([t1_step] * 2)
            
            # embeddings = torch.cat([self.embeddings['pos'][:,None].repeat(1,batch_size, 1, 1), 
                                    # self.embeddings['neg'][:,None].repeat(1,batch_size, 1, 1)])
            embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

            noise_pred_t1 = self.unet(
                latent_model_input_t1, tt1, encoder_hidden_states=embeddings
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_cond, noise_pred_uncond = noise_pred_t1.chunk(2)
            noise_pred_t1 = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # [batch,4,64,64]   sigma_t1:[batch,1,1,1]
            di = (latents_noisy_t1 - noise_pred_t1) / sigma_t1
            latents_noisy_t2 = latents_noisy_t1 + (sigma_t2 - sigma_t1)*di
            # latents_noisy_t2 = latents_noisy_t2.unsqueeze(0)
            latent_model_input_t2 = torch.cat([latents_noisy_t2] * 2)
            # latent_model_input_t2 = latent_model_input_t1 + (sigma_t2 - sigma_t1)*di
            
            # 这里要加 sg()
            x0_pred_sub_t1 = (noise - di)
            noise_pred_t2 = self.unet(
                latent_model_input_t2, tt2, encoder_hidden_states=embeddings
            ).sample
            noise_pred_cond, noise_pred_uncond = noise_pred_t2.chunk(2)
            noise_pred_t2 = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            # noise_pred_t2 = noise_pred_t2[0]

            # w(t), sigma_t^2
            w_t2 = (1 - self.alphas[t2_step]).view(batch_size, 1, 1, 1)

            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)


        x0_pred = self.add_noise(latents, x0_pred_sub_t1, sigma_t1)
        loss = (w_t2 * F.mse_loss(x0_pred, noise_pred_t2.detach(), reduction='sum')).sum()

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)
 
                x0_pred = self.decode_latents(x0_pred.to(latents.type(self.dtype)))

                latents_recon_t2input = self.predict_start_from_noise_cds(latents_noisy_t2, t2, noise_pred_t2)
                result_hopefully_less_noisy_image_t2input = self.decode_latents(latents_recon_t2input.to(latents.type(self.dtype)))
                latents_recon_t1input = self.predict_start_from_noise_cds(latents_noisy_t1, t1, noise_pred_t1)
                result_hopefully_less_noisy_image_t1input = self.decode_latents(latents_recon_t1input.to(latents.type(self.dtype)))
                
                latents_recon_ddpm_intot2 = self.predict_start_from_noise(latents_noisy_t2, t2_step, noise_pred_t2)
                result_hopefully_less_noisy_image_ddpm_intot2 = self.decode_latents(latents_recon_ddpm_intot2.to(latents.type(self.dtype)))
                
                latents_recon_ddpm_intot1 = self.predict_start_from_noise(latents_noisy_t1, t1_step, noise_pred_t1)
                result_hopefully_less_noisy_image_ddpm_intot1 = self.decode_latents(latents_recon_ddpm_intot1.to(latents.type(self.dtype)))
                
                # visualize noisier image
                # result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.dtype))

                # TODO: also denoise all-the-way

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                # viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                viz_images = torch.cat([pred_rgb_512, result_hopefully_less_noisy_image_t2input, result_hopefully_less_noisy_image_t1input,
                                        x0_pred,
                                        result_hopefully_less_noisy_image_ddpm_intot2, result_hopefully_less_noisy_image_ddpm_intot1],dim=0)
                save_image(viz_images, os.path.join(save_guidance_path,"guidance.png"))
                print("t2step:",t2_step, "t1_step", t1_step)

        return loss

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

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
