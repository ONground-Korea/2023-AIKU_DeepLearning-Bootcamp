# Curriculum
## 1. Machine Learning and MLP  
## 2. CNN and CNN Architectures
- ### Assignment #2
    - Implementation of AlexNet and VGG16
## 3. Object Detection and Segmentation
- ### Table of Contents
    ```py
    1. Object Detection
        1.1) Detection a single object
        1.2) Detecting Multiple Objects: Sliding Window
        1.3) Region Proposals
    2. R-CNN: Region-Based CNN
        2.1) Train-time
        2.2) Test-time
        2.3) Comparing Boxes: Intersection over Union (IoU)
        2.4) Overlapping Boxes: Non-Max Suppression (NMS)
        2.5) Evaluating Object Detectors: Mean Average Precision (mAP)
    3. Fast R-CNN
        3.1) 설명
        3.2) Cropping Features: RoI Pool
        3.3) Cropping Features: RoI Align
        3.4) Fast R-CNN vs "Slow" R-CNN
    4. Faster R-CNN: Learnable Region Proposals
        4.1) 설명
        4.2) Region Proposal Network (RPN)
        4.3) 4 Losses
        4.4) 성능
        4.5) Single-Stage Object Detection
    5. Object Detection: Lots of variables!
        5.1) Open-Source Code
    Summary
    ```
    
    ```py
    1. Training R-CNN
        1.1) "Slow" R-CNN
        1.2) Fast R-CNN
        1.3) Faster R-CNN
    2. Cropping Features
        2.1) RoI Align
    3. Detection without Anchors: CornerNet
    4. Semantic Segmentation
        4.1) Semantic Segmentation Idea: Sliding Window
        4.2) Semantic Segmentation: Fully Convolutional Network
        4.3) In-Network Upsampling: “Unpooling”
        4.4) In-Network Upsampling: Bilinear Interpolation
        4.5) In-Network Upsampling: Bicubic Interpolation
        4.6) In-Network Upsampling: “Max Unpooling”
        4.7) Learnable Upsampling: Transposed Convolution
    5. Instance Segmentation
        5.1) Computer Vision Tasks
        5.2) Computer Vision Tasks: Instance Segmentation
        5.3) Mask R-CNN
    6. Panoptic Segmentation
        6.1) Beyond Instance Segmentation
        6.2) Beyond Instance Segmentation: Panoptic Segmentation
    7. Human Keypoints
        7.1) Mask R-CNN
        7.2) Joint Instance Segmentation and Pose Estimation
    8. General Idea: Add PerRegion “Heads” to Faster / Mask R-CNN!
        8.1) Dense Captioning: Predict a caption per region!
        8.2) 3D Shape Prediction: Predict a 3D triangle mesh per region!
    Summary
    ```

- ### Assignment #3
    - Implement Custom Dataset and Finetune Faster RCNN
## 4. RNN and Attention
- ### Table of Contents
    ```py
    1. Process Sequences
        1.1) Sequential Processing of Non-Sequential Data
    2. Recurrent Neural Networks
        2.1) (Vanilla) Recurrent Neural Networks
        2.2) RNN Computational Graph
        2.3) Sequence to Sequence(seq2seq)
        2.4) Truncated Backpropagation Through Time
        2.5) Examples
        2.6) Searching for Interpretable Hidden Units
    3. Image Captioning
    4. Attention(cs231n_2017)
    5. VQA(cs231n_2017)
    6. Vanilla RNN Gradient Flow
    7. LSTM
        7.1) 모델 설명
        7.2) 과정 설명
        7.3) LSTM Gradient Flow
    ```
    ```py
    1. Sequence-to-Sequence with RNNs
        1.1) 문제점 및 해결방안
    2. Sequence-to-Sequence with RNNs and "Attention"
        2.1) 설명
        2.2) 예시
        2.3) 정리
    3. Image Captioning with RNNs and Attention
        3.1) 설명
        3.2) 예시
    4. Attention Layer
        4.1) 설명
        4.2) Similarity function
        4.3) Multiple Query vectors 
        4.4) Separate Input vectors into key and value
        4.5) 요약
    5. Self-Attention Layer
        5.1) 설명
        5.2) Input의 순서를 바꾼다면?
    6. Masked Self-Attention Layer
    7. Multihead Self-Attention Layer
    8. Example: CNN with Self-Attention
    9. Three Ways of Processing Sequences
    10. Transformer
        10.1) 설명
        10.2) Transfer Learning
    Summary
    ```
## 5. Natural Language Processing
## 6. 3D Vision and Videos
- ### Table of Contents

    ```py
    1. 3D Vision topics
    2. 3D Shape Representations  
        2.1) 3D Shape Representations: Depth Map
        2.2) 3D Shape Representations: Surface Normals
        2.3) 3D Shape Representations: Voxels
        2.4) Scaling Voxels
        2.5) 3D Shape Representations: Implicit Functions
        2.6) 3D Shape Representations: Point Cloud
        2.7) 3D Shape Representations: Triangle Mesh
    3. Shape Comparison Metrics
        3.1) Shape Comparison Metrics: Intersection over Union
        3.2) Shape Comparison Metrics: Chamfer Distance
        3.3) Shape Comparison Metrics: F1 Score (Best)
        3.4) Summary
    4. Cameras
        4.1) Cameras: Canonical vs View Coordinates
        4.2) View-Centric Voxel Predictions
    5. 3D Datasets
        5.1) 3D Datasets: Object-Centric
    6. 3D Shape Prediction: Mesh R-CNN
        6.1) Mesh R-CNN: Hybrid 3D shape representation
        6.2) Mesh R-CNN Pipeline
        6.3) Mesh R-CNN Results
    ```
    ```py
    1. Video Classification
        1.1) Problem: Videos are big!
        1.2) Training on Clips
        1.3) Video Classification(1): Single-Frame CNN
        1.4) Video Classification(2): Late Fusion (with FC layers)
        1.5) Video Classification(2): Late Fusion (with pooling)
        1.6) Video Classification: Early Fusion
        1.7) Video Classification: 3D CNN
    2. Early Fusion vs Late Fusion vs 3D CNN
    3. 2D Conv (Early Fusion) vs 3D Conv (3D CNN)
        3.1) 2D Conv (Early Fusion)
        3.2) 3D Conv (3D CNN)
        3.3) Performance : Early Fusion vs Late Fusion vs 3D CNN
        3.4) C3D: The VGG of 3D CNNs
    4. Recognizing Actions from Motion
        4.1) Measuring Motion: Optical Flow
    ```
## 7. Generative Models
- ### Table of Contents

    ```py
    1. Supervised Learning vs Unsupervised Learning
    2. Discriminative vs Generative Models
        2.1) 위 Models을 가지고 무엇을 할 수 있을까
        2.2) Taxonomy of Generative Models
    3. Autoregressive Models
        3.1) Explicit Density Estimation
        3.2) Explicit Density: Autoregressive Models
        3.3) PixelRNN
        3.4) PixelCNN
        3.5) 결과 및 결론
    4. Variational Autoencoders
        4.1) (Regular, non-variational) Autoencoders
        4.2) Variational Autoencoders Overview
        4.3) Variational Autoencoders
    ```
    ```py
    1. Variational Autoencoders
        1.1) VAE : Train
        1.2) VAE : Generating Data
        1.3) VAE : Editing with z
        1.4) VAE : Summary
        1.5) Autoregressive models vs Variational models
    2. Vector-Quantized Variational Autoencoder (VQ-VAE2)
    3. Autoregressive model, VAE, GANs Overview
    4. Generative Adversarial Networks
        4.1) GANs : Training Objective
        4.2) GANs : Optimality
        4.3) GANs : DC-GAN
        4.4) GANs : Vector Math
        4.5) GAN Improvments
    5. Conditional GANs
        5.1) Conditional Batch Normalization
        5.2) Spectral Normalization
        5.3) Self-Attention
        5.4) BigGAN
        5.5) Generating Videos with GANs
        5.6) Conditioning on more than labels! Text to Image
        5.7) Image Super-Resolution: Low-Res to High-Res
        5.8) Image-to-Image Translation: Pix2Pix
        5.9) Unpaired Image-to-Image Translation: CycleGAN
        5.10) Label Map to Image
        5.11) GANs: Not just for images! Trajectory Prediction
    6. Generative Models Summary
    ```

- ### Assignment #7
    - VAE and Improving VAE