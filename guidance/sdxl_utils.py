from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    AutoPipelineForInpainting
)
from diffusers.utils.import_utils import is_xformers_available


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms

from diffusers.utils import load_image

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
class StableDiffusionXL(nn.Module):
    def __init__(
        self,
        device,
        fp16=False,
        vram_O=False,
        hf_key=None,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device

        

        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        self.pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
                                                         torch_dtype=self.dtype,
                                                         local_files_only=True
                                                         ).to(self.device)
        self.generator = torch.Generator(device="cuda").manual_seed(0)


        # if vram_O:
        #     pipe.enable_sequential_cpu_offload()
        #     pipe.enable_vae_slicing()
        #     pipe.unet.to(memory_format=torch.channels_last)
        #     pipe.enable_attention_slicing(1)
        #     # pipe.enable_model_cpu_offload()
        # else:
        #     pipe.to(device)

        # self.vae = pipe.vae
        # self.tokenizer = pipe.tokenizer
        # self.text_encoder = pipe.text_encoder
        # self.unet = pipe.unet

        # self.scheduler = DDIMScheduler.from_pretrained(
        #     model_key, subfolder="scheduler", torch_dtype=self.dtype, local_files_only=True
        # )

        # del pipe

        # self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        # self.min_step = int(self.num_train_timesteps * t_range[0])
        # self.max_step = int(self.num_train_timesteps * t_range[1])
        # self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        # self.embeddings = {}
    
    @torch.no_grad()
    def inpaint(self, 
                pred_rgbs,
                masks,
                prompt,
                negative_prompt=None,
                step_ratio=None,
                num_inference_steps = 20,
                strength=0.99,
                guidance_scale=100,
                as_latent=False,
                guidance_img_path=None):
        
        pred_rgb_1024 = F.interpolate(pred_rgbs, (1024, 1024), mode='bilinear', align_corners=False)
        
        masks_float = masks.float()
        masks_1024 = F.interpolate(masks_float, (1024, 1024), mode='bilinear', align_corners=False)
        # masks_1024 = masks_1024 > 0.5

        
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pred_rgb_1024,
            mask_image=masks_1024,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,  # steps between 15 and 30 work well for us
            strength=strength,  # make sure to use `strength` below 1.0
            generator=self.generator,
            )
        
        transform = transforms.ToTensor()
        inpainted_images = []
        for img in out.images:
            inpainted_images.append(transform(img).unsqueeze(0))    
        inpainted_images = torch.cat(inpainted_images, dim=0).to(self.device)
        
        if guidance_img_path:
            # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
            # viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
            masks_expand = masks_1024.expand_as(pred_rgb_1024)
            
            # reshape
            pred_small = F.interpolate(pred_rgb_1024, (256, 256), mode='bilinear', align_corners=False)
            masks_small = F.interpolate(masks_expand, (256, 256), mode='bilinear', align_corners=False)
            inpainted_small = F.interpolate(inpainted_images, (256, 256), mode='bilinear', align_corners=False)

            viz_images = torch.cat([pred_small, masks_small, inpainted_small],dim=0)
            # viz_noise_images = torch.cat([latents[0], imgs],dim=0)
            save_image(viz_images, guidance_img_path)
        
        return inpainted_images




if __name__ == "__main__":
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float32,
        local_files_only=True
        ).to("cuda")

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    image = load_image(img_url).resize((1024, 1024))
    mask_image = load_image(mask_url).resize((1024, 1024))

    # prompt = "a tiger sitting on a park bench"
    prompt = "a dog sitting on a park bench"
    generator = torch.Generator(device="cuda").manual_seed(0)

    final_image = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    guidance_scale=8.0,
    num_inference_steps=20,  # steps between 15 and 30 work well for us
    strength=0.99,  # make sure to use `strength` below 1.0
    generator=generator,
    ).images[0]
    # cat image and mask_image and final_image
    final_image.save("a.png")
