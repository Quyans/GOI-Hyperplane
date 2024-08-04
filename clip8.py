import torch
import cv2
import argparse

def downsample():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, default='input.png', help='input image')
    # parser.add_argument('--output', type=str, default='output.png', help='output image')
    # args = parser.parse_args()
    for i in range(0, 1622, 4):
        image = cv2.imread(f'../datasets/scannet/scans/scene0241_02/image/color/{i}.jpg')
        image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
        cv2.imwrite(f'../datasets/nets/{i}.jpg', image)
    print('downsample x4 done')

def main():
    import clip
    image_features = torch.load('../datasets/scannet/scans/scene0241_02/label/f68.pt')
    h, w = 121, 162
    print(image_features.shape)
    # fclip, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_e16')
    # fclip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
    # clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)

    # text = fclip_tokenizer(['chair', 'desk', 'wall', 'window', 'floor', 'lamp', 'shelf'])
    # text_features = fclip.encode_text(text).type(torch.float32).cuda()
    clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)

    text = clip.tokenize(['chair', 'desk', 'wall', 'window', 'floor', 'lamp', 'shelf']).cuda()
    text_features = clip_pretrained.encode_text(text).type(torch.float32)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print(text_features.shape)
    # different color for each label, image features are (C, h, w)
    image_features = image_features.permute(1, 2, 0).reshape(-1, 512)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    # color green blue red orange
    new_img = torch.zeros(h * w, 3, dtype=torch.uint8, device='cuda')
    pallete = torch.tensor([[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255], [255,165,0],[2,165,128],[92,75,128]], dtype=torch.uint8, device='cuda')
    for i in range(h * w):
        # cos_sim = torch.cosine_similarity(image_features[i].unsqueeze(0), text_features)
        # argmax = torch.argmax(cos_sim)
        # if cos_sim[argmax] > 0.8:
        #     new_img[i] = pallete[argmax]
        dot_sim = image_features[i] @ text_features.t()
        argmax = torch.argmax(dot_sim)
        new_img[i] = pallete[argmax]
    new_img = new_img.reshape(h, w, 3)
    print(new_img.shape)
    # save image
    import cv2
    cv2.imwrite('../datasets/nno/t1.png', new_img.cpu().numpy())
    print('done')


def scale():
    for i in range(656, 935):
        pt = torch.load(f'../datasets/kitchen/img/DSCF0{i}.pt')
        pad = torch.nn.functional.interpolate(pt.unsqueeze(0), scale_factor=0.25, mode='bilinear')
        pad = pad.squeeze(0)
        # pad = torch.nn.functional.pad(pt,(1,1,0,0),'replicate')
        # down x2
        # pad = torch.nn.functional.interpolate(pad, scale_factor=0.5, mode='bilinear')
        torch.save(pad, f'../datasets/kitchen/clip_feat/DSCF0{i}.pt')
        if i == 1: print(pt.shape, pad.shape)
    # pt = torch.load('../datasets/nno/0.pt')
    # pad = torch.nn.functional.pad(pt,(2,2,0,0),'replicate')
    print(pt.shape, pad.shape)

if __name__ == '__main__':
    # main()
    # pt = torch.load(f'../datasets/book_store/img/frame_00001.pt')
    # print(pt.shape, pt.dtype)
    # downsample()
    scale()
