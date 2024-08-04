
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np

namelist = ["test_latent"]
for name in namelist:
    
    tensor = torch.load(os.path.join('testfolder', name + '.pt')).cpu()
    
    # 可视化
    rgb_image = tensor[0:3, :, :]
    # rgb_image = rgb_image + 1
    # rgb_image = rgb_image/2 

    # 转换Tensor以符合matplotlib的期望格式[H, W, C]
    rgb_image = rgb_image.permute(1, 2, 0).to(torch.float)
    H, W = rgb_image.shape[0], rgb_image.shape[1]
    # 确保数据在0到1的范围内
    rgb_image = torch.clamp(rgb_image, 0, 1)

    rgb_image_255 = (rgb_image * 255).type(torch.uint8)

    image = Image.fromarray(rgb_image_255.numpy())

    # 保存图像
    image.save(os.path.join("/home/quyansong/Project/LE-Gaussian/data/exp", name + '_lantent.png'))
    
    # pca to 3
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    
    pca_data = torch.clamp(tensor, 0, 1)
    q = pca_data.permute(1,2,0).clone().reshape(-1, 10).cpu().numpy()
    q = pca.fit_transform(q)
    q = q.reshape(H, W, 3)
    cv2.imwrite("/home/quyansong/Project/LE-Gaussian/data/exp/" +str(name) + "_pca".png, (q * 255).astype(np.uint8))




# # 可视化RGB图像
# plt.imshow(rgb_image)
# plt.title("RGB Image from Tensor Channels")
# plt.axis('off')  # 关闭坐标轴
# plt.show()



# plt.imshow(selected_channel_data, cmap='gray')
# plt.title(f"Channel {channel}")
# plt.colorbar()
# plt.show()