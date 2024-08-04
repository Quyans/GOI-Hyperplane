import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
from math import atan, degrees, tan, radians
import trimesh
import os
def visualize_poses(poses, size=0.1,save_dir="logs/", file_name="test.glb"):
    # poses: [B, 4, 4]
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for i in range(len(poses)):
        pose = poses[i]
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    # trimesh.Scene(objects).show()
    trimesh.Scene(objects).export(os.path.join(save_dir,file_name))

def visualize_poses_stabledreamfusion(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()

def get_c2w_with_RT(R,T):
    
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt).astype(np.float32)
    return C2W

def intrinsic_to_fov(f_x, f_y, width, height):
    """
    根据相机内参计算视场角。

    参数:
    f_x (float): x轴的焦距（像素单位）
    f_y (float): y轴的焦距（像素单位）
    width (int): 图像宽度（像素单位）
    height (int): 图像高度（像素单位）

    返回:
    (float, float): (水平视场角, 垂直视场角)
    """
    fov_x = 2 * degrees(atan(width / (2 * f_x)))  # 水平视场角
    fov_y = 2 * degrees(atan(height / (2 * f_y)))  # 垂直视场角

    return fov_x, fov_y

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T


class OrbitCamera:
    def __init__(self, W, H, r=1, fovy=60,fovx=None, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.fovx = np.deg2rad(fovx) if fovx is not None else 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        
    
    # @property
    # def fovx(self):
    #     return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    def set_pose(self, c2w):
        if isinstance(c2w, np.ndarray):
            self.T = c2w
        else:
            self.T = c2w.numpy()
    
    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy, dz=0):
        # rotate along camera up/side axis!
        # qy, qx = -self.rot.as_matrix()[:3, 1], self.rot.as_matrix()[:3, 0]
        qx, qy, qz = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])

        rotvec_x = qx * np.radians(-1.5*dy)
        rotvec_y = qy * np.radians(-1.5*dx)
        rotvec_z = qz * np.radians(dz)
        rot_xt = R.from_rotvec(rotvec_z) * R.from_rotvec(rotvec_y) * R.from_rotvec(rotvec_x)
        self.rot = self.rot * rot_xt
        # self.center = -rot_xt.apply(-self.center)

    def scale(self, delta):
        if self.radius==0:
            self.radius = 1
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])

    def import_pose(self, pose):
        if isinstance(pose, np.ndarray):
            None
        else:
            pose = pose.detach().cpu().numpy()
        self.center = -pose[:3, 3]
        self.rot = R.from_matrix(pose[:3, :3])
        # 为了保证完全对齐radius
        self.radius = 0
        
    def get_cam_location(self):
        return self.pose[:3,3]
    
 
    @staticmethod
    # def rand_poses(device, batch, elevation_range=[0,30], azimuth_range=[-180, 180], radius=1, is_degree=True):
    def rand_poses(batch=10, elevation_range=[0,0], azimuth_range=[0, 360], radius=1, is_degree=True, rotate_z=False):
            
        elevation_range = np.array(elevation_range) / 180 * np.pi
        azimuth_range = np.array(azimuth_range) / 180 * np.pi

        elevation = np.random.rand(batch) * (elevation_range[1] - elevation_range[0]) + elevation_range[0]
        azimuth = np.random.rand(batch) * (azimuth_range[1] - azimuth_range[0]) + azimuth_range[0]

        
        centers = np.zeros((batch, 3))#【B,3】
        # centers = np.array([0, 0, 0])  # [3]

        target_x = radius * np.cos(elevation) * np.sin(azimuth)
        target_y = radius * np.sin(elevation)
        target_z = radius * np.cos(elevation) * np.cos(azimuth)        
        # 在NumPy中，unsqueeze操作可以用np.newaxis来替代
        target_x = target_x[:, np.newaxis]  # 从[batch] 扩展为[batch, 1]
        target_y = target_y[:, np.newaxis]
        target_z = target_z[:, np.newaxis]

        target = np.concatenate((target_x, target_y, target_z), axis=1)
        
        forward_vector = safe_normalize(target - centers)
        up_vector = np.tile(np.array([0, 1, 0]), (batch, 1))
        right_vector = safe_normalize(np.cross(forward_vector, up_vector, axis=-1))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector, axis=-1))

        # 这样得到的是w2c， 可视化可视的应该是c2w!!!  look_at方法应该是得到的是w2c
        # 即right_vector, up_vector, forward_vector分别在第一行第二行第三行，而不是第一列第二列第三列 所以cat的时候dim=1
        poses = np.tile(np.eye(4), (batch, 1, 1))
        poses[:, :3, :3] = np.stack((right_vector, up_vector, forward_vector), axis=1)
        
        translation = np.array([
            [1, 0, 0, -centers[0, 0]],
            [0, 1, 0, -centers[0, 1]],
            [0, 0, 1, -centers[0, 2]],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        translation = np.tile(translation, (batch, 1, 1))
        poses = np.matmul(poses, translation)
        
        if rotate_z:
            poses[:, :, 0] *= -1
            poses[:, :, 1] *= -1

        
        # back to degree
        elevation = elevation / np.pi * 180
        azimuth = azimuth / np.pi * 180
        # return poses, elevation, azimuth
        
        c2w = np.linalg.inv(poses)
        c2w = np.float32(c2w)
        return c2w, elevation, azimuth
