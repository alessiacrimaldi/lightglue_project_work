# Image Processing and Computer Vision Porject Work

## Homography estimation with LightGlue: Experiment Reproduction

**Goal**: Replicate the homography estimation experiment from the Lightglue paper using the HPatches dataset.

**Implementation Details**: This project evaluates the robustness and accuracy of LightGlue's transformer-based local feature matcher by estimating homographies between image pairs and comparing them with ground truth. Key components include keypoint detection (`SuperPoint`), feature matching (`LightGlue`), homography computation via `RANSAC/MAGSAC` and `DLT`, and quantitative evaluation using reprojection error. The setup follows the original paper's protocol, with direct benchmarking and visualization of matching quality under viewpoint and illumination changes.

### 📂 HPatches Dataset

The HPatches dataset is a widely adopted dataset for evaluating feature matching and homography estimation. Its structure makes it ideal for reproducible benchmarking:
- Consists of sequences of 5 image pairs, each under either illumination or viewpoint changes: two of the most challenging scenarios in real-world image matching.
- Provides ground truth homographies, allowing for accurate and objective evaluation of homography estimation performance.

### 🖼️ Example Images

<p align="center"> 
  <img src="./media/image_pair.png" alt="Challenging image pair under viewpoint and illumination changes used for the test" width="100%" /> 
  <br>
  <em>Figure 1: Challenging image pair under viewpoint and illumination changes used for the test</em> 
</p> 
<p align="center"> 
  <img src="./media/matches.png" alt="Result of the feature extraction and matching using SuperPoint and LightGlue" width="100%" /> 
  <br>
  <em>Figure 2: Result of the feature extraction and matching using SuperPoint and LightGlue</em> 
</p> 
<p align="center"> 
  <img src="./media/image_warping.png" alt="Result of the homography estimation using LightGlue's feature matching" width="60%" />
  <br>
  <em>Figure 3: Result of the homography estimation using LightGlue's feature matching</em> 
</p>

### 📖 References

- [LightGlue: Local Feature Matching at Light Speed](https://arxiv.org/abs/2306.16472), Philipp Lindenberger, Paul-Edouard Sarlin, Marc Pollefeys. ICCV 2023.
- [HPatches Dataset: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors](https://github.com/hpatches/hpatches-dataset?tab=readme-ov-file), Vassileios Balntas, Karel Lenc, Andrea Vedaldi, Krystian Mikolajczyk. CVPR, 2017.
