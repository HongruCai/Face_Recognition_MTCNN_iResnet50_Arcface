# Face\_Recognition\_MTCNN\_iResnet50\_Arcface

Face recognition and annotation based on MTCNN, iResNet50, and Arcface

Author: Hongru Cai

## 📄 Project Introduction
This project implements a face recognition and annotation system using MTCNN, iResNet50, and ArcfaceLoss.
It supports both image-based and video-based face recognition with annotation, aiming to provide a reproducible pipeline for face detection, feature extraction, and identity matching.

Although the current implementation is functional, several components can be further optimized for speed, accuracy, and robustness. This repository is intended as a research-oriented reference for those exploring deep learning–based face recognition.

📬 For inquiries, please contact: henry.hongrucai@gmail.com

## 📂 File Structure

```plaintext
Face_Recognition_MTCNN_iResnet50_Arcface
├─ Application_Pictures_Video/        # Application: Image & video annotation
│   ├─ Picture_Recognition.py         # Image annotation
│   ├─ Video_Recognition.py           # Video annotation
│   ├─ facebank/                      # Face database
│   ├─ face_input/                    # Raw images or videos
│   └─ model/                         # Model files
│       ├─ iResnet.py                 # iResNet model
│       └─ params/                    # Model parameters
│
├─ Train_Test_Modules/                # Train & Test: Model training and testing
│   ├─ metrics.py                     # ArcfaceLoss function
│   ├─ test_module.py                  # Model testing
│   └─ train_module.py                 # Model training
│
├─ data/                              # Training and testing datasets
│
├─ MTCNN_Module/                      # MTCNN module
│   ├─ MTCNN.py                       # Face detection & cropping
│   └─ src/                           # Functions for MTCNN
│       ├─ box_utils.py
│       ├─ detector.py
│       ├─ first_stage.py
│       ├─ get_nets.py
│       ├─ visualization_utils.py
│       ├─ __init__.py
│       └─ weights/                   # MTCNN model parameters
│
├─ model/                             # Model files
│   ├─ fmobilenet.py                  # MobileFaceNet model
│   └─ iResnet.py                     # iResNet model
│
└─ params/                            # Parameters for training & testing
```

## 🏋️ Model Training

* Training script: `train_module.py`
* **Preparation**:

  * Download a dataset (JPG format recommended) and place it in `data/`.
  * If faces are not cropped, use **MTCNN** to preprocess them.
  * Modify the dataset path in `train_module.py`.
  * For pretrained weights, place them in `params/` and update the model path.
  * Adjust parameters like batch size, optimizer, and loss as needed.
* **Training steps**:

  1. Input images into the network to extract features.
  2. Compute **ArcfaceLoss** (cosine distance) with labels.
  3. Calculate loss and backpropagate.
* **Notes**:

  * On small datasets, loss can drop to \~0.5.
  * Overfitting may occur.
  * If `nan` values appear, check data integrity.

## 🧪 Model Testing

* Testing script: `test_module.py`
* **Method**: Compare similarity between two faces in the **LFW dataset**.
* **Reference**: [Build-Your-Own-Face-Model](https://github.com/siriusdemon/Build-Your-Own-Face-Model/blob/master/recognition/blog/test.md)
* **Results**:

  * Custom-trained models: \~95% accuracy.
  * [insight\_face](https://github.com/deepinsight/insightface/tree/master/recognition) official model: \~98% accuracy.

## 🖼️ Image Face Annotation

* Script: `Picture_Recognition.py`
* **Preparation**:

  * Place reference faces in `face_bank/`.
  * Place target images in `face_input/`.
  * Update image paths in the script.
  * Place model weights in `params/`.
  * Adjustable parameters:

    * **Face size threshold**: default `50` px.
    * **Distance threshold**: default `0.228` (LFW optimal).
    * **Draw landmarks**: default `True`.
* **Workflow**:

  1. Detect and crop faces using **MTCNN**.
  2. Compare extracted features with `face_bank`.
  3. Assign the label with the closest match or mark as `unknown`.
  4. Annotate bounding boxes, labels, and landmarks.
  5. Save results and clean intermediate files.

## 🎥 Video Face Annotation

* Script: `Video_Recognition.py`
* **Workflow**:

  1. Extract frames from the video.
  2. Perform annotation per frame (same as image process).
  3. Recombine annotated frames into a video.
* **Notes**:

  * Works well for MP4; other formats may cause errors.
  * Large frame counts increase processing time.

## 📚 References

* [arcface\_torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
* [InsightFace\_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
* [pytorch\_arcface\_cosface\_partialFC](https://github.com/leoluopy/pytorch_arcface_cosface_partialFC)
* [Manual Implementation of arcface\_torch in insightFace](https://zhuanlan.zhihu.com/p/368510746)
* [Face Recognition Series | 10 ArcFace Analysis](https://zhuanlan.zhihu.com/p/76541084)

