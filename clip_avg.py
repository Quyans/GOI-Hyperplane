import torch
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image
import argparse
import os
import numpy as np
from tqdm import tqdm


# usage: python clip_avg.py -i ../datasets/360/bicycle -o /ssd/dsh/clip_feat/bike

class Patcher:
    """Class to patch and unpatch data.
    Adopted from https://github.com/EmilienDupont/coinpp.

    Args:
        patch_shape (tuple of ints). Patch size of audio, image or volume. For example
            (200,) for audio, (64, 64) for an image or (32, 32, 16) for a volume.

    Notes:
        Only works for regular volumetric data, such as images and MRI scans,
        not spherical data such as ERA5.
    """

    def __init__(self, patch_shape):
        assert len(patch_shape) in (1, 2, 3)
        self.patch_shape = patch_shape
        self.patch_dims = len(patch_shape)
        # self.is_3d = len(patch_shape) == 3

    def patch(self, data):
        """Splits data into patches. If the patch shape doesn't divide the data
        shape, use reflection padding.

        Args:
            data (torch.Tensor): Shape (channels, width) or (channels, height, width) or
                (channels, depth, height, width). Note that there should not be
                a batch dimension.

        Returns:
            Patched data of shape (num_patches, channels, {depth, height,} width)
            and a tuple ({depth, height,} width) specifiying the original shape
            of the data (this is required to reconstruct the data).
        """
        assert data.ndim == 3, "Incorrect data shape for images."

        # Extract shapes
        channels = data.shape[0]
        spatial_shape = data.shape[1:]
        patch_height, patch_width = self.patch_shape

        # Pad data so it can be divided into equally sized patches
        pad_height, pad_width = self.get_padding(spatial_shape, self.patch_shape)
        # Note that padding operates from last to first in terms of dimension
        # i.e. (left, right, top, bottom)
        padding = (0, pad_width, 0, pad_height)
        padded = torch.nn.functional.pad(data, padding, mode="reflect")

        # padded has shape (channels, padded_height, padded_width),
        # unsqueeze this to add a batch dimension (expected by unfold)
        patches = torch.nn.functional.unfold(
            padded.unsqueeze(0),
            stride=self.patch_shape,
            kernel_size=self.patch_shape,
        )
        # patches has shape (1, channels * patch_height * patch_width, num_patches).
        # Reshape to (num_patches, channels, patch_height, patch_width)
        patches = patches.reshape(channels, patch_height, patch_width, -1).permute(
            3, 0, 1, 2
        )
        # Return patches and data shape, so data can be reconstructed from
        # patches
        return patches, spatial_shape

    def unpatch(self, patches, spatial_shape):
        """
        Args:
            patches (torch.Tensor): Shape (num_patches, channels, {patch_depth,
                patch_height,} patch_width).
            spatial_shape (tuple of ints): Tuple describing spatial dims of
                original unpatched data, i.e. ({depth, height,} width).
        """
        # Calculate padded shape (required by fold function)
        height, width = spatial_shape
        pad_height, pad_width = self.get_padding(spatial_shape, self.patch_shape)
        padded_shape = (height + pad_height, width + pad_width)

        # Reshape patches tensor from (num_patches, channels, patch_height, patch_width)
        # to (1, channels * patch_height * patch_width, num_patches)
        num_patches, channels, patch_height, patch_width = patches.shape
        patches = patches.permute(1, 2, 3, 0).reshape(1, -1, num_patches)
        # Fold data to return a tensor of shape (1, channels, padded_height, padded_width)
        padded_data = torch.nn.functional.fold(
            patches,
            output_size=padded_shape,
            kernel_size=self.patch_shape,
            stride=self.patch_shape,
        )

        # Remove padding to get tensor of shape (channels, height, width)
        return padded_data[0:, :, :height, :width]

    @staticmethod
    def get_padding(spatial_shape, patch_shape):
        """Returns padding required to make patch_shape divide data_shape into equal
        patches.

        Args:
            spatial_shape (tuple of ints): Shape ({depth, height,} width).
            patch_shape (tuple of ints): Shape ({patch_depth, patch_height,} patch_width).
        """
        patch_height, patch_width = patch_shape
        height, width = spatial_shape
        excess_height = height % patch_height
        excess_width = width % patch_width
        pad_height = patch_height - excess_height if excess_height else 0
        pad_width = patch_width - excess_width if excess_width else 0
        return pad_height, pad_width


class ClipFeatPyramid:
    def __init__(self,
                 image_size=(1024, 1024),
                 num_level=5,
                 out_compress=True):
        short_size = min(image_size)
        self.patchers = []
        self.num_level = num_level
        self.compress = out_compress
        self.clip, _ = clip.load('ViT-B/32', device='cuda', jit=False)
        self.preprocess = Compose([
            Resize(self.clip.visual.input_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(self.clip.visual.input_resolution),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        for i in range(1, num_level + 1):
            stride = 2 ** i
            patch_size = short_size // stride
            patcher = Patcher((patch_size, patch_size))
            self.patchers.append(patcher)

    def forward(self, data):
        feats = []
        for i, patcher in enumerate(self.patchers):
            patches, img_shape = patcher.patch(data)
            pw, ph = patches.shape[2:]
            patches = self.preprocess(patches)
            with torch.no_grad():
                feat = self.clip.encode_image(patches)
            # repeat [B,3] -> [B,C,pw,ph]
            feat = feat.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, pw, ph)
            whole_feat = patcher.unpatch(feat, img_shape)
            feats.append(whole_feat)  # [1,C,W,H]
        lvl0 = self.preprocess(data.unsqueeze(0))
        with torch.no_grad():
            feat0 = self.clip.encode_image(lvl0).unsqueeze(-1).unsqueeze(-1)
        flat_feat = torch.cat(feats, dim=0)
        pic_feat = (flat_feat.sum(0) + feat0) / (self.num_level + 1)
        if self.compress:
            pic_feat = torch.nn.functional.interpolate(pic_feat, scale_factor=0.5, mode='bilinear')
        return pic_feat.squeeze(0).half().cpu()  # [C,W,H]


def test():
    img = ToTensor()(Image.open("DSCF4698.JPG")).float().cuda()
    clip_feat = ClipFeatPyramid(
        image_size=img.shape[1:],
        num_level=5
    )
    q = clip_feat.forward(img).permute(1, 2, 0).float()
    r = torch.rand(512, 3).cuda() / 7
    s = q @ r
    s = s.clip(0, 1) * 255
    import cv2
    cv2.imwrite("DS28.png", s.cpu().numpy())
    # torch.save(q, "DSCF4698.pt")


def img2feat(args):
    img_folder = args.img
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)
    clip_feat = None
    for img_name in tqdm(os.listdir(img_folder)):
        img = ToTensor()(Image.open(os.path.join(img_folder, img_name))).float().cuda()
        if clip_feat is None:
            clip_feat = ClipFeatPyramid(
                image_size=img.shape[1:],
                num_level=args.num_levels
            )
        feat = clip_feat.forward(img)
        basename = os.path.basename(img_name).split(".")[0]
        torch.save(feat, os.path.join(output_folder, basename + ".pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", '-i', type=str, help="input image folder path")
    parser.add_argument("--output", '-o', type=str, help="output feature folder path")
    parser.add_argument("--num_levels", '-l', type=int, default=5, help="Feature levels")
    arguments = parser.parse_args()

    img2feat(arguments)
