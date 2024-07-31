
<p align="center">
<!--   <h1 align="center"><img height="100" src="https://github.com/imlixinyang/director3d-page/raw/master/assets/icon.ico"></h1> -->
  <h1 align="center">🆒 <b>GOI</b>: Find 3D Gaussians of Interest with an Optimizable Open-vocabulary Semantic-space Hyperplane</h1>
  <p align="center">
        <a href="https://arxiv.org/abs/2405.17596"><img src='https://img.shields.io/badge/arXiv-GOI-red?logo=arxiv' alt='Paper PDF'></a>
        <a href='https://quyans.github.io/GOI-Hyperplane/'><img src='https://img.shields.io/badge/Project_Page-GOI-green' alt='Project Page'></a>
        <a href='https://drive.google.com/file/d/1JunEiWyPNwGprdqXh-D2dTQPFMaRisDz/view?usp=sharing'><img src='https://img.shields.io/badge/Dataset-MipNeRF360 OV-yellow?logo=databricks' alt='Project Page'></a>
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

Visiting our [**Project Page**](https://goi-hyperplane.github.io/) for more result.

## 🔧 Installation
- clone this repo:
```
git clone https://github.com/Quyans/GOI-Hyperplane.git
cd GOI-Hyperplane
```

- set up a new conda environment
```
conda env create --file environment.yml
conda activate goi
```

## 📚 Data Preparation
We use datasets in the COLMAP format. For your own dataset, you can use the convert.py script. Refer to [3DGS](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) for specific usage.

In addition to RGB data, we also use pixel-aligned semantic feature maps. Specifically, we use APE as our vision-language model to extract semantic features from images.

First, install our modified [APE repository](https://github.com/Atrovast/APE). Then, run the following command to extract semantic features from all RGB images and save them in the `clip_feat` folder under the scene path:
```shell
cd ../APE
python demo/demo_lazy.py -i <scene_path>/images/* --feat-out <scene_path>/clip_feat/
```

After preparing the depth maps, your scene folder should look like this:
```
scene_path
├── clip_feat/
├── images/
├── sparse/
└── ...
```

## 🚋 Training
To reconstruct the 3D scene along with the semantic features, run the following command:
```shell
python train.py -s <scene path> -m <model path> -i <alternative image path>
```

Ensure that the resolution of the feature maps matches the resolution of the RGB images. For example, if you're using images from the `images_4` folder to extract semantic features, use the `-i images_4` option in the `train.py` script.

For detailed usage instructions for `train.py`, please refer to the [3DGS documentation](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#running).

## 👀 Visualization

After completing the reconstruction, you can visualize the results using our GUI. 

First, download the language model of APE from [here](), and place it in the `models` folder in the root directory.

To start the GUI, run the following command:
```shell
python gui/main_test.py --config gui/configs/config_test.yaml
```
Note: A few additional models will be automatically downloaded the first time you run the script.


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
