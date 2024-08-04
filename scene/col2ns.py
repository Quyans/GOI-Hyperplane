import os
import sys
import numpy as np
import json
from PIL import Image
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary


def readColmapSceneInfo(path, scale: int = 1):
    scale = int(scale)
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if scale == 1 else f'images_{scale}'
    camera_list = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), alt=reading_dir, scale=scale)
    with open(os.path.join(path, "transforms.json"), 'w', encoding='utf-8') as f:
        json.dump(camera_list, f, indent=4)


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, alt, scale):
    cam_dict = {
        'camera_model': 'OPENCV',
        'orientation_override': 'none',
    }
    frame_list = []

    scene1 = 'room'
    l1 = os.listdir(f'/home/dsh/datasets/ovm360/{scene1}')
    print(l1)
    l2 = [os.listdir(os.path.join(f'/home/dsh/datasets/ovm360/{scene1}', i, 'images')) for i in l1]
    l3 = [y.split('.')[0]  for x in l2 for y in x]
    train_list, test_list = [], []
    sample_img = os.path.join(images_folder, os.listdir(images_folder)[0])
    height, width = np.array(Image.open(sample_img)).shape[:2]

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        # height = intr.height
        # width = intr.width

        uid = intr.id
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        if True in np.isnan(R) or True in np.isnan(T):
            print("Math error in camera {}!".format(idx))
            continue
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        c2w = np.linalg.inv(w2c)
        # opencv_c2w is the same as colmap
        c2w[:3, 1:3] *= -1

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = focal_length_y = intr.params[0]
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        base_name = os.path.basename(extr.name)
        path_image = os.path.join('./', alt, base_name)
        if base_name.split('.')[0] not in l3:
            train_list.append(path_image)
        else:
            test_list.append(path_image)

        frame = {
            "fl_x": focal_length_x / scale,
            "fl_y": focal_length_y / scale,
            "cx": intr.params[2] / scale,
            "cy": intr.params[3] / scale,
            "w": width,
            "h": height,
            "file_path": path_image,
            "transform_matrix": c2w.tolist()
        }
        frame_list.append(frame)

        # image_path = os.path.join(images_folder, os.path.basename(extr.name))
        # image_name = os.path.basename(image_path)
        # image = Image.open(image_path)

        # ape_path = os.path.join(images_folder, f'../clip_feat/{image_name}.pt')
        # clip_path = os.path.join(images_folder, f'../clip_avg/{image_name}.pt')
        # ape_feat = torch.load(ape_path).cpu()
        # clip_feat = torch.load(clip_path).cpu()
        # semantic = {'ape': ape_feat, 'clip': clip_feat}

        # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
        #                       image_path=image_path, image_name=image_name, width=width, height=height,
        #                       semantic=semantic, semantic_path=ape_path)
        # cam_infos.append(cam_info)
    sys.stdout.write('\n')
    cam_dict['frames'] = frame_list
    cam_dict['train_filenames'] = train_list
    cam_dict['test_filenames'] = test_list
    cam_dict['val_filenames'] = []
    return cam_dict

# {
#     "camera_model": "OPENCV",
#     "orientation_override": "none",
#     "frames": [
#         {
#             "fl_x": 775.0429534770752,
#             "fl_y": 775.0429534770752,
#             "cx": 498.0262451171875,
#             "cy": 343.7258605957031,
#             "w": 994,
#             "h": 738,
#             "file_path": "./images/frame_00001.jpg",
#             "transform_matrix":


if __name__ == "__main__":
    readColmapSceneInfo(sys.argv[1], sys.argv[2])
