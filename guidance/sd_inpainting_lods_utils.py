
from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline
)
from diffusers.utils.import_utils import is_xformers_available


import numpy as np
import torch
import torch.nn as nn
import PIL
import torch.nn.functional as F
from torchvision.utils import save_image

import requests
from io import BytesIO
from PIL import Image
import os
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

def pil_to_numpy(images) -> np.ndarray:
    """
    Convert a PIL image or a list of PIL images to NumPy arrays.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=False,
        vram_O=False,
        hf_key=None,
        t_range=[0.02, 0.98],
        grad_clip=10,
    ):
        super().__init__()

        self.device = device

        
        model_key = "stabilityai/stable-diffusion-2-inpainting"

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
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}
        self.learnable_uncond_text_embeddings = None
        self.optimizer_embedding = None
        
        self.grad_clip = grad_clip
        

    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts).to(dtype=self.dtype)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts).to(dtype=self.dtype)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds
        return self.embeddings   
    
    def init_embedding_optimizer(self, lr=1e-3):    
        self.learnable_uncond_text_embeddings = nn.Parameter(self.embeddings['neg'].detach().clone())
        self.optimizer_embeddings = torch.optim.Adam([self.learnable_uncond_text_embeddings], lr=lr)

    def get_embedding_optimizer(self):
        return self.optimizer_embeddings
    
    @torch.no_grad()
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

    def train_embedding(
        self,
        latents,
        mask_latent,
        masked_image_latents,
        uncond_text_embeddings,
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
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        latent_model_input = torch.cat([latents_noisy, mask_latent, masked_image_latents], dim=1)

        noise_pred_text = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=uncond_text_embeddings,
        ).sample
        # 这里是不是也应该是mask级别的mse呢？
        return F.mse_loss(noise_pred_text, noise.detach(), reduction="mean")
    
    
    def train_step(
        self,
        pred_rgb,
        masks,
        masks_nodilated=None,
        step_ratio=None,
        guidance_scale=7.5,
        x0_guidance=False,
        vers=None, hors=None,
        save_guidance_path=None
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)
        masks = masks.to(self.dtype)

        # interp to 512x512 to be fed into vae.
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
        # 对mask插值并且二值化
        masks_512 = F.interpolate(masks, (512, 512), mode="bilinear", align_corners=False)
        masks_512[masks_512 < 0.5] = 0
        masks_512[masks_512 >= 0.5] = 1
        
        if masks_nodilated is not None:
            masks_nodilated_512 = F.interpolate(masks_nodilated, (512, 512), mode="bilinear", align_corners=False)
            masks_nodilated_512[masks_nodilated_512 < 0.5] = 0
            masks_nodilated_512[masks_nodilated_512 >= 0.5] = 1
        
        masked_image_latents = self.encode_masked_imgs(pred_rgb_512, masks_512)
        
        mask_latent = torch.nn.functional.interpolate(
            masks_512, size=(512 // self.vae_scale_factor, 512 // self.vae_scale_factor)
        )
        mask_latent = mask_latent.to(device=self.device, dtype=self.dtype)
        
        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_512)

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            
            # t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = np.round( (step_ratio) **1 * (self.min_step - self.max_step) + self.max_step )
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        text_embeds = self.embeddings['pos'].repeat(batch_size, 1, 1)
        uncond_text_embeddings = self.learnable_uncond_text_embeddings.repeat(batch_size, 1, 1)

        embeddings = torch.cat([text_embeds, uncond_text_embeddings])
        
        with torch.no_grad():
            
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents_noisy] * 2)
            mask_latent_t = torch.cat([mask_latent] * 2)
            masked_image_latents_t = (
                torch.cat([masked_image_latents] * 2)
            )
            # concat latents, mask, masked_image_latents_t in the channel dimension
            latent_model_input = torch.cat([latent_model_input, mask_latent_t, masked_image_latents_t], dim=1)
            
            tt = torch.cat([t] * 2)

            noise_pred = self.unet(
                latent_model_input, tt, encoder_hidden_states=embeddings
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            # original sds
            # noise_pred = noise_pred_uncond + guidance_scale * (
            #     noise_pred_cond - noise_pred_uncond
            # )
            
            # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
            noise_pred =  noise_pred_cond + (1 - guidance_scale) / guidance_scale * noise_pred_uncond - noise / guidance_scale

            grad = w * noise_pred
            grad = torch.nan_to_num(grad)
            grad = grad.clamp(-self.grad_clip, self.grad_clip)
            
            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)
        x0_mse = False
        if x0_mse:
            # (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
            
            alpha_prod_t = self.alphas[t].view(batch_size, 1, 1, 1)
            
            beta_prod_t = 1 - alpha_prod_t
            target = ((latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)).detach()
        else:
            target = (latents - grad).detach()
            
        x0_guidance = False
        ## 在2d上做sds
        if x0_guidance:
            latents_x0 = self.decode_latents(latents)
            target_x0 = self.decode_latents(target)
            
            mse_loss = nn.MSELoss(reduction='none')
            mse_loss = 0.5 * mse_loss(latents_x0.float(), target_x0) / latents_x0.shape[0]
            masks_expanded = masks.expand_as(target_x0)   # [b, 1, H, W] -> [b, 4, H, W]
            mse_loss = mse_loss * masks_expanded
            loss = mse_loss.sum()
        else:
            # loss = 0.5 * F.mse_loss(latents.float() * mask_latent, target, reduction='sum') / latents.shape[0]
            mse_loss = nn.MSELoss(reduction='none') #能手动应用mask
            if x0_mse:
                mse_loss = mse_loss(latents.float(), target)
            else:
                
                mse_loss = 0.5 * mse_loss(latents.float(), target)/ latents.shape[0]
                
                # semantic_masks [1,1,512,512] 而 rgb是[1,3,512,512] 因此需要扩张
                masks_latent_expanded = mask_latent.expand_as(latents)   # [b, 1, 64, 64] -> [b, 4, 64, 64]
                mse_loss = mse_loss * masks_latent_expanded
                loss_sds = mse_loss.sum()
                
                # loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]
                loss_embedding = self.train_embedding(latents, mask_latent, masked_image_latents, uncond_text_embeddings)
                
                if save_guidance_path:
                    with torch.no_grad():
                        latents_recon = self.preditc_start_from_noise(latents_noisy[0], t, noise_pred)
                        result_hopefully_less_noisy_image = self.decode_latents(latents_recon.to(latents.type(self.dtype)))

                        # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                        masks_512 = masks_512.expand_as(pred_rgb_512) 
                        
                        if masks_nodilated is not None:
                            masks_nodilated_512 = masks_nodilated_512.expand_as(pred_rgb_512)
                            viz_images = torch.cat([pred_rgb_512, masks_512,masks_nodilated_512 ,result_hopefully_less_noisy_image],dim=0)
                        else:
                            viz_images = torch.cat([pred_rgb_512, masks_512 ,result_hopefully_less_noisy_image],dim=0)                
                        save_image(viz_images, save_guidance_path)
                        print(t)
                

                return loss_sds, loss_embedding
            
            
            
            
            # masks_expanded = torch.nn.functional.interpolate(
            #     masks_latent_expanded[:,0:3,...], size=(masks_latent_expanded.shape[-2] * self.vae_scale_factor, masks_latent_expanded.shape[-1] * self.vae_scale_factor)
            # )

       
        
        return loss

    @torch.no_grad()
    def produce_latents(
        self,
        image,
        mask_image,
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
                    self.vae.config.latent_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
                dtype=self.dtype
            )

        latents = latents * self.scheduler.init_noise_sigma  #这个感觉可有可无
        
        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])


        masks_512 = F.interpolate(mask_image, (512, 512), mode="bilinear", align_corners=False)
        masks_512[masks_512 < 0.5] = 0
        masks_512[masks_512 >= 0.5] = 1
        
        # normalize image to [-1, 1]
        masked_image_latents = self.encode_masked_imgs(image, masks_512)
        
        mask_latent = torch.nn.functional.interpolate(
            masks_512, size=(512 // self.vae_scale_factor, 512 // self.vae_scale_factor)
        )
        mask_latent = mask_latent.to(device=self.device, dtype=self.dtype)

        mask_latent_t = torch.cat([mask_latent] * 2)
        masked_image_latents_t = (
            torch.cat([masked_image_latents] * 2)
        )

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            
            latent_model_input = torch.cat([latent_model_input, mask_latent_t, masked_image_latents_t], dim=1)
            # t = t.unsqueeze(-1)
            # tt = torch.cat([t] * 2).to(self.device)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings,timestep_cond=None,cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # # compute the previous noisy sample x_t -> x_t-1
            # latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents , return_dict=False)[0]
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1   #中心化，将像素值映射到[-1, 1]  即(imgs-0.5)*2  但是这个再diffuser中也有做 就是normallize

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
    
    def encode_masked_imgs(self,image, mask):
        # imgs: [B, 3, H, W]
        image = 2 * image - 1
        masked_image = image * (mask<0.5)
        masked_image = masked_image.to(device=self.device, dtype=self.dtype)

        # 与上面这个相比 因为要masked 所以应该先标准化再mask 而不是先mask再标准化，这样原本的0就成-1了
        posterior = self.vae.encode(masked_image).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
    
    def preditc_start_from_noise(self, latents_noisy, t, noise_pred):
        # latents_noisy是 latents 加噪之后的xt
        total_timesteps = self.max_step - self.min_step + 1
        index = total_timesteps - t.to(latents_noisy.device) - 1 
        b = len(noise_pred)
        a_t = self.alphas[index].reshape(b,1,1,1).to(self.device) #a_t 即alpha_t
        sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas)
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
        latents_recon = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
        return latents_recon
        
    def prompt_to_inpainting(self, prompt, negative_prompt, 
                             image,
                             mask_image,
                             steps,
                             latents=None,
                             ):
        if isinstance(prompt, str):
            prompts = [prompt]

        if isinstance(negative_prompt, str):
            negative_prompts = [negative_prompt]
        
        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        
        # Text embeds -> img latents
        latents = self.produce_latents(
            image,
            mask_image,
            height=512,
            width=512,
            latents=latents,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        

        return imgs.detach()

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    # parser.add_argument("--prompt",default="a corgi sitting on a bench", type=str)
    parser.add_argument("--prompt",default="a desk", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=133)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()
    
    # img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    # mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    # image = download_image(img_url).resize((512, 512))
    # # image
    # mask_image = download_image(mask_url).resize((512, 512))
    # # mask_image
    # prompt = "a corgi sitting on a bench"

    image = Image.open("/home/quyansong/Project/LE-Gaussian/guidance/pred_rgb_512.png").resize((512, 512))
    mask_image = Image.open("/home/quyansong/Project/LE-Gaussian/guidance/masks_512_dilated.png").resize((512, 512))
    prompt = opt.prompt
    
    guidance_scale=15 #7.5
    num_samples = 5
    generator = torch.Generator(device="cuda").manual_seed(133) # change the seed to get different results

    use_pipe = True
    if use_pipe:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        pipe.to("cuda")
        
        images = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_samples,
        ).images
        images.insert(0, image)
        def save_image_grid(image_grid, filename):
            """Saves the image grid as a PNG image file."""
            image_grid.save(filename, format="PNG")  # You can change format to "JPEG" if needed

        # After creating the image grid:
        grid = image_grid(images, 1, num_samples + 1)

        # image_grid(images, 1, num_samples + 1)
        save_image_grid(grid, "inpainted_results.png")  # Change filename as desired
        # image_grid.save("test", format="PNG")
    else: 
        # 使用自己的代码
        seed_everything(opt.seed)
        
        # 处理一下image和maskimage  PIL.IMAGE -> torch.tensor
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255


        mask_image = mask_image.convert("L") # convert to grayscale
        mask_image = [np.array(mask_image).astype(np.float32) / 255.0]
        mask_image = np.stack(mask_image, axis=0)
        mask_image = mask_image[..., None]
        
        mask_image = torch.from_numpy(mask_image.transpose(0, 3, 1, 2))

        image_tensor = image_tensor.to("cuda")
        mask_image = mask_image.to("cuda")
        sd = StableDiffusion(device="cuda", fp16=False, vram_O=False, hf_key=None, t_range=[0.02, 0.98])
        images = sd.prompt_to_inpainting(opt.prompt, opt.negative, image_tensor, mask_image, opt.steps)
        # Img to Numpy
        imgs = images.cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")
        
        # numpy to pil
        if imgs.ndim == 3:
            imgs = imgs[None, ...]
        if imgs.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in imgs]
        else:
            pil_images = [Image.fromarray(image) for image in imgs]
                        
        pil_images.insert(0, image)
        def save_image_grid(image_grid, filename):
            """Saves the image grid as a PNG image file."""
            image_grid.save(filename, format="PNG")  # You can change format to "JPEG" if needed

        grid = image_grid(pil_images, 1, len(pil_images))

        save_image_grid(grid, "inpainted_mycode.png")  # Change filename as desired

    
    
