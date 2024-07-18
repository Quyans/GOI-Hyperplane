
<p align="center">
<!--   <h1 align="center"><img height="100" src="https://github.com/imlixinyang/director3d-page/raw/master/assets/icon.ico"></h1> -->
  <h1 align="center">🆒 <b>GOI</b>: Find 3D Gaussians of Interest with an Optimizable Open-vocabulary Semantic-space Hyperplane</h1>
  <p align="center">
        <a href="https://arxiv.org/abs/2405.17596"><img src='https://img.shields.io/badge/arXiv-GOI-red?logo=arxiv' alt='Paper PDF'></a>
        <a href='https://goi-hyperplane.github.io/'><img src='https://img.shields.io/badge/Project_Page-GOI-green' alt='Project Page'></a>
        <a href=''><img src='https://img.shields.io/badge/Dataset-GOI-yellow?logo=databricks' alt='Project Page'></a>
  </p>

<!-- <img src='assets/pipeline.gif'> -->
**😊 TL;DR**

GOI can locate 3D gaussians of interests as directed by open-vocabulary prompts.


**⭐ Key components of GOI**:


- A Trainable Feature Clustering Codebook effciently condense noisy high-dimensional semantic features into compact, low-dimensional vectores, ensuring well-defined segmentation boundaries.
- Finetuning Semantic-space Hyperplane, initiallized by text query embedding, to better locate target area.
- An open-vocabulary dataset is proposed, named MipNeRF360-OV.

**🔥 News**:

- 🥰 Check out our new gradio demo by simply running ```python app.py```.


## 📖 Open-vocabulary Query Results
❗ You can precisely locate 3D Gaussians of Interest with an open-vocabulary text prompt


https://github.com/user-attachments/assets/dd392b1e-3acc-4745-a4f4-940d5d3f44b0

<img src='assets/teaser.gif'>

Visiting our [**Project Page**](https://goi-hyperplane.github.io/) for more result.

## 🔧 Installation
- create a new conda enviroment

```
conda create -n goi python=3.10
conda activate goi
```

- install pytorch (or use your own if it is compatible with ```xformers```)
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
- install ```xformers``` for momory-efficient attention
```
conda install xformers -c xformers
```
- install ```pip``` packages
```
pip install kiui scipy opencv-python-headless kornia omegaconf imageio imageio-ffmpeg  seaborn==0.12.0 plyfile ninja tqdm diffusers transformers accelerate timm einops matplotlib plotly typing argparse gradio kaleido==0.1.0.post1
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install "git+https://github.com/ashawkey/diff-gaussian-rasterization.git"
```

- clone this repo:
```
git clone https://github.com/Quyans/GOI-Hyperplane.git
cd GOI-Hyperplane
```

<!-- - download the pre-trained model by:
```
wget https://huggingface.co/imlixinyang/director3d/resolve/main/model.ckpt?download=true -O model.ckpt
``` -->

## Citation

```
@article{goi2024,
    title={GOI: Find 3D Gaussians of Interest with an Optimizable Open-vocabulary Semantic-space Hyperplane},
    author={Qu, Yansong and Dai, Shaohui and Li, Xinyang and Lin, Jianghang and Cao, Liujuan and Zhang, Shengchuan and Ji, Rongrong},
    journal={arXiv preprint arXiv:2405.17596},
    year={2024}
}
```


## License

Licensed under the CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)


The code is released for academic research use only. 

If you have any questions, please contact me via [quyans@stu.xmu.edu.cn](mailto:quyans@stu.xmu.edu.cn). 
