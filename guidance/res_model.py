import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import subprocess, io, os, sys, time

# groundingDINO_path_fromguidance = os.path.join(os.path.dirname(__file__), '../ext')
# groundingDINO_path = os.path.join(os.path.dirname(__file__), 'ext')
# sys.path.append(groundingDINO_path)
# sys.path.append(groundingDINO_path_fromguidance)

sys.path.append("../GOI-Hyperplane")
sys.path.append("../GOI-Hyperplane/ext")
sys.path.append("../GOI-Hyperplane/ext/GroundingDINO")


from torchvision import transforms

from torchvision.transforms import ToPILImage
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from huggingface_hub import hf_hub_download

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
import copy
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import clip

from utils.image_utils import transform_Image_to_tensor

import matplotlib
matplotlib.use('AGG')
plt = matplotlib.pyplot

lama_cleaner_enable = True
if lama_cleaner_enable:
    try:    
        from lama_cleaner.model_manager import ModelManager
        from lama_cleaner.schema import Config as lama_Config    
    except Exception as e:
        lama_cleaner_enable = False

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
    args = SLConfig.fromfile(model_config_path) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model    

def get_sam_vit_h_4b8939():
    if not os.path.exists(sam_checkpoint):
        result = subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'], check=True)

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)

        try:
            font = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
            font_size = 36
            new_font = ImageFont.truetype(font, font_size)

            draw.text((x0+2, y0+2), str(label), font=new_font, fill="white")
        except Exception as e:
            pass

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask



config_file = "./ext/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
sam_checkpoint = './sam_vit_h_4b8939.pth' 

class RES_MODEL(nn.Module):
    def __init__(
        self,
        device,
        fp16=False
    ):
        super().__init__()

        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32
        
        self.load_groundingdino_model(self.device)
        self.load_sam_model(self.device)
        
        self.clip_model, preprocess = clip.load("ViT-B/32", device=self.device, jit=False)

        
    
    def load_groundingdino_model(self, device):
    # initialize groundingdino model
        print(f"initialize groundingdino model...")
        groundingdino_model = load_model_hf(config_file, ckpt_repo_id, ckpt_filenmae, device=device) #'cpu')
        self.groundingdino_model = groundingdino_model
        
    def load_sam_model(self, device):
        # initialize SAM
        global sam_model, sam_predictor, sam_mask_generator, sam_device
        get_sam_vit_h_4b8939()
        sam_device = device
        self.sam_model = build_sam(checkpoint=sam_checkpoint).to(sam_device)
        self.sam_predictor = SamPredictor(self.sam_model)
        self.sam_mask_generator = SamAutomaticMaskGenerator(self.sam_model)

    def load_lama_cleaner_model(self, device):

        self.lama_cleaner_model = ModelManager(
                name='lama',
                device=device,
            )

    # 对应于grounded-SAM的 load_image()
    def transform_image(self, image):
        # # load image
        if isinstance(image, PIL.Image.Image):
            image_pil = image
        elif torch.is_tensor(image):
            # image shape [3,H,W]
            to_pil_image = ToPILImage()
            image_pil = to_pil_image(image)
        else:
            print("image type not supported")
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases
    
    def run_anything_task(self, input_image, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, 
            iou_threshold, inpaint_mode, mask_source_radio, remove_mode, remove_mask_extend, num_relation, kosmos_input, cleaner_size_limit=1080):

        text_prompt = text_prompt.strip()

        file_temp = int(time.time())
        
        output_images = []

        # load image
        image_pil, image = self.transform_image(input_image)
        input_img = input_image
        output_images.append(input_image)

        size = image_pil.size
        H, W = size[1], size[0]

        # run grounding dino model
    
        # groundingdino_device = 'cpu'
        # if device != 'cpu':
        #     try:
        #         from groundingdino import _C
        #         groundingdino_device = 'cuda:0'
        #     except:
        #         warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only in groundingdino!")

        boxes_filt, pred_phrases = self.get_grounding_output(
            self.groundingdino_model, image, text_prompt, box_threshold, text_threshold, device=self.device
        )
        
        if boxes_filt.size(0) == 0:
            return [], None, None, None

        boxes_filt_ori = copy.deepcopy(boxes_filt)

        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }

        image_with_box = plot_boxes_to_image(copy.deepcopy(image_pil), pred_dict)[0]
        output_images.append(image_with_box)

        if task_type == 'segment':
            # [C,H,W] -> [H,W,C]
            image = (np.array(input_img).transpose(1,2,0) *255).astype(np.uint8)
            if self.sam_predictor:
                self.sam_predictor.set_image(image)

            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            if self.sam_predictor:
                boxes_filt = boxes_filt.to(sam_device)
                transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

                # masks, _, _, _ = self.sam_predictor.predict_torch(
                masks, _, _= self.sam_predictor.predict_torch(  
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )
                # masks: [9, 1, 512, 512]
                assert sam_checkpoint, 'sam_checkpoint is not found!'
            else:
                masks = torch.zeros(len(boxes_filt), 1, H, W)   
                mask_count = 0         
                for box in boxes_filt:
                    masks[mask_count, 0, int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1  
                    mask_count += 1   
                masks = torch.where(masks > 0, True, False)      
                run_mode = "rectangle"

            # draw output image

            
            # draw output image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                show_box(box.cpu().numpy(), plt.gca(), label)
            plt.axis('off')
            
            output_dir = "./"
            image_path = os.path.join(output_dir, f"grounding_seg_output.jpg")
            plt.savefig(image_path, bbox_inches="tight")
            plt.clf()
            plt.close('all')

            segment_image_result = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            os.remove(image_path)
            output_images.append(Image.fromarray(segment_image_result)) 
  
            return  output_images, masks, boxes_filt, pred_phrases


    def text_similarity(self, text1, text2):
        text = clip.tokenize([text1, text2]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
        similarity = torch.cosine_similarity(text_features[0], text_features[1], dim=-1).item()
        return similarity
    
    def predict_res_mask(self, image, text_prompt):
        # image tensor 【3，h, w】
        
        # w, h = image.size 
        h, w = image.shape[1:]
        
        run_rets = self.run_anything_task(input_image = image, 
                                        text_prompt = text_prompt,  
                                        task_type = 'segment', 
                                        inpaint_prompt = '', 
                                        box_threshold = 0.3, 
                                        text_threshold = 0.25, 
                                        iou_threshold = 0.8, 
                                        inpaint_mode = "merge", 
                                        mask_source_radio = "type what to detect below", 
                                        remove_mode = "rectangle",   # ["segment", "rectangle"]
                                        remove_mask_extend = "10", 
                                        num_relation = 5,
                                        kosmos_input = None,
                                        cleaner_size_limit = -1,
                                        )
        
        images_pil = run_rets[0]
        # transform = transforms.ToTensor()
        # output_image = transform(images_pil[0])
        # output_image = transform_Image_to_tensor(images_pil[0])
        output_image = images_pil[0]
        masks, boxes, pred = run_rets[-3:]
        masks = masks.cpu().to(torch.float32)
        boxes = boxes.cpu()

        if pred and len(pred) > 0:
            # prob = [float(x.split('(')[-1][:-1]) for x in pred]
            prob = [self.text_similarity(text_prompt, x.split('(')[0]) for x in pred]
            prob_ind_des = np.argsort(prob)[::-1]
            for i in range(1, len(prob_ind_des)):
                if prob[prob_ind_des[i]] < 0.99 * prob[prob_ind_des[0]] or prob[prob_ind_des[i]] < 0.9 * prob[prob_ind_des[i-1]]:
                    prob_ind_des = prob_ind_des[:i]
                    break
            masks = [masks[i] for i in prob_ind_des]
            boxes = [boxes[i] for i in prob_ind_des] 
            
            # 这个排序没什么必要
            pred = [pred[i] for i in prob_ind_des]
            prob = [float(x.split('(')[-1][:-1]) for x in pred]
            prob_ind_des = np.argsort(prob)[::-1]
            for i in range(1, len(prob_ind_des)):
                if prob[prob_ind_des[i]] < 0.8 * prob[prob_ind_des[0]] or prob[prob_ind_des[i]] < 0.8 * prob[prob_ind_des[i-1]]:
                    prob_ind_des = prob_ind_des[:i]
                    break
            masks = [masks[i] for i in prob_ind_des]
            boxes = [boxes[i] for i in prob_ind_des]

            mask_stack = torch.stack(masks, dim=0)
            mask_out = torch.sum(mask_stack, dim=0).clamp(0, 1) if len(masks) > 0 else torch.zeros((h, w)).to(self.device)            
            
            # cv2.imwrite("masktest.png", ma_np*255)
        else:
            mask_out = torch.zeros((1,h,w))

        return mask_out, output_image
        
        
        
if __name__=="__main__":
    
    model = RES_MODEL(device='cuda')
    image = Image.open('/ssd/qys/Project/LE-Gaussian/guidance/test_img/pred_rgb_512.png')
    text_prompt = 'the sofa chair'
    img_tensor = transform_Image_to_tensor(image)
    mask_out, output_image = model.predict_res_mask(img_tensor, text_prompt)
    
    mask_out_expand = mask_out.expand_as(output_image) 
    viz_images = torch.cat([output_image.unsqueeze(0) ,mask_out_expand.unsqueeze(0)],dim=0)                
    save_image(viz_images, "masktest.png")