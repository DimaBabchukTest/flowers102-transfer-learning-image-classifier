# Flowers102 Image Classification with Transfer Learning

## Executive Summary

This project implements a **multi-class image classification system** for the Flowers102 dataset using **transfer learning with lightweight convolutional neural networks**. The objective is to evaluate how well modern pre-trained models generalize under **severe data constraints**, while applying **senior-level best practices** in model training, evaluation, and deployment readiness.

Three architectures were evaluated: **MobileNetV3-Small**, **MobileNetV3-Large**, and **EfficientNet-B0**. These models were intentionally selected because they combine **strong ImageNet-pretrained representations** with **computational efficiency**, making them suitable for both **server-side inference** and **potential edge/mobile deployment**.

A key conclusion of this work is:

> **Deep learning models are fundamentally data-hungry. When labeled data is limited, overfitting becomes a structural limitation rather than a modeling error.**

---

## Problem Statement

- **Task:** Multi-class image classification (102 flower categories)
- **Dataset:** Flowers102
- **Challenges:**
  - Only ~10 training images per class
  - Fine-grained visual differences between categories
  - High risk of overfitting
- **Objectives:**
  - Compare lightweight pre-trained CNN architectures
  - Apply principled overfitting mitigation techniques
  - Select the best model using robust, statistically sound evaluation

---

## Dataset Overview

This project uses the **Oxford Flowers 102** dataset, a widely used benchmark for fine-grained image classification.

🔗 **Dataset reference:**  
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

The dataset contains images of **102 flower species**, with large visual similarity between classes, making it a challenging classification task.

---
## EDA Summary

### Executive Summary

This exploratory data analysis (EDA) examines the structure, size, class distribution, and visual characteristics of the Flowers102 dataset prior to model development. The analysis confirms that while the training and validation splits are perfectly balanced, the test split is substantially larger and exhibits significant class imbalance. These findings directly motivate the choice of macro-averaged evaluation metrics and influence how model performance is interpreted.

---

### Dataset Size and Split Characteristics

The **Flowers102 dataset contains a total of 8,189 images** spanning **102 flower categories**, using official predefined splits:

- **Training set:** 1,020 images (10 images per class)
- **Validation set:** 1,020 images (10 images per class)
- **Test set:** 6,149 images (variable number of images per class)

The training and validation sets are intentionally small and uniformly balanced, while the test set is significantly larger and more representative of a real-world, non-uniform class distribution.


### Class Distribution Analysis

A quantitative analysis of label frequencies reveals the following:

- **Training set**
  - Classes: 102  
  - Min images per class: 10  
  - Max images per class: 10  
  - Max/Min ratio: 1.0  
  - Perfectly balanced  

- **Validation set**
  - Classes: 102  
  - Min images per class: 10  
  - Max images per class: 10  
  - Max/Min ratio: 1.0  
  - Perfectly balanced  

- **Test set**
  - Classes: 102  
  - Min images per class: 20  
  - Max images per class: 238  
  - Max/Min ratio: 11.9  
  - Mean images per class: ~60  

Bar plot visualizations of class distributions clearly confirm that **only the test set is imbalanced**, while the training and validation splits are strictly uniform.


### Visual Inspection of Sample Images

To better understand the visual complexity of the dataset, representative images from different classes were inspected:

- Nine sample images from distinct flower classes were visualized
- Images were denormalized and displayed alongside their corresponding class names
- The samples reveal:
  - High intra-class variability
  - Strong inter-class visual similarity
  - Subtle differences in color, texture, and structure

This confirms that Flowers102 represents a **fine-grained image classification problem**, where small visual cues can significantly impact model predictions.


### Key Insights and Modeling Implications

- **No imbalance correction is required during training**, as the training and validation sets are perfectly balanced.
- **Test set imbalance must be handled at evaluation time**, not by altering the data.
- Standard accuracy can be misleading under this test-set distribution.
- **Macro-averaged metrics** (Macro-F1, Macro-AUC, Macro-Precision, Macro-Recall) are therefore essential to ensure fair, class-balanced evaluation.
- The combination of **small training size** and **fine-grained visual differences** significantly increases the risk of overfitting.

Overall, the EDA confirms that Flowers102 is a challenging dataset well suited for studying **generalization limits, overfitting behavior, and the data requirements of transfer learning models** under low-data regimes.

---
## Model Selection Rationale

The following ImageNet-pretrained architectures were evaluated:

- **MobileNetV3-Small**
- **MobileNetV3-Large**
- **EfficientNet-B0**

These models were chosen because they:

- Provide **high-quality learned representations** despite relatively small parameter counts
- Are **lightweight and efficient**, suitable for real-world deployment constraints
- Offer distinct accuracy–efficiency trade-offs:
  - *MobileNetV3-Small:* edge / mobile-friendly inference
  - *MobileNetV3-Large:* higher-capacity server-side inference
  - *EfficientNet-B0:* balanced efficiency and accuracy
- Adapt well to **transfer learning under limited data**

This selection reflects a **practical engineering mindset**, rather than reliance on heavyweight benchmark models.

---

## Training Strategy & Best Practices

The training pipeline follows established best practices in applied deep learning, with a strong emphasis on **reproducibility**, **controlled experimentation**, and **robust model selection**:

- **Reproducible training setup**
  - Fixed random seeds across Python, NumPy, and PyTorch
  - Deterministic backend settings where applicable
  - Version-pinned dependencies to ensure consistent results across environments

- **Two-stage fine-tuning**
  - Warm-up phase with a frozen pretrained backbone
  - Full fine-tuning with a reduced learning rate to adapt high-level features

- **Hyperparameter tuning for regularization**
  - Systematic tuning of **dropout rates** and **weight decay** values
  - Separate tuning per model architecture to account for capacity differences
  - Selection based on validation performance rather than training metrics

- **Validation-loss–based checkpointing**
  - Model checkpoints saved based on the lowest validation loss
  - Prevents selecting models that are already overfitting

- **Early stopping**
  - Training halted when validation loss stops improving
  - Reduces unnecessary training and limits overfitting

- **Regularization techniques**
  - Dropout to reduce co-adaptation of features
  - Weight decay to constrain model complexity

- **Strong data augmentation**
  - Geometric and color-based augmentations
  - Increases effective data diversity under limited data conditions

The test set is never used during training, hyperparameter tuning, or checkpoint selection.

---

## Evaluation Methodology

### Evaluation Metrics

Model performance is evaluated using macro-averaged metrics appropriate for multi-class classification:

- Accuracy  
- **Macro-AUC**
- **Macro-F1**
- Macro-Precision
- Macro-Recall

Macro-averaging ensures that each class contributes equally, which is critical given the imbalanced nature of the test set.

### Statistical Robustness

A robust evaluation methodology was employed:

- Custom functions were implemented to compute advanced macro-averaged metrics
- **Bootstrapping** was used to estimate **95% Confidence Intervals (CIs)** for all metrics
- Model comparisons were based on CI overlap:
  - **Non-overlapping CIs** suggest statistically meaningful differences
  - **Overlapping CIs** indicate that observed differences may not be statistically reliable
- Key metrics (Macro-AUC and Macro-F1) were visualized with **error bars representing 95% CIs**

---

## Final Results & Model Selection

### Model Performance Summary (After Regularization)

Three models were evaluated after hyperparameter tuning and regularization.

#### **MobileNetV3-Large (Dropout = 0.3, Weight Decay = 1e-05)**

- **Accuracy:** 0.8520 (95% CI: 0.8434 – 0.8605)
- **Macro-AUC:** 0.9971 (95% CI: 0.9966 – 0.9975)
- **Macro-F1:** 0.8459 (95% CI: 0.8345 – 0.8534)
- **Macro-Precision:** 0.8319 (95% CI: 0.8218 – 0.8412)
- **Macro-Recall:** 0.8772 (95% CI: 0.8693 – 0.8855)

#### **EfficientNet-B0 (Dropout = 0.2, Weight Decay = 1e-05)**

- Competitive performance across all metrics
- Generally second-best model

#### **MobileNetV3-Small (Dropout = 0.2, Weight Decay = 1e-05)**

- Lowest performance among the three models
- Most constrained by model capacity

---

### Statistical Significance Analysis

- **MobileNetV3-Large** was **statistically significantly better** than **MobileNetV3-Small** across *all metrics*
- **MobileNetV3-Large** was **statistically significantly better** than **EfficientNet-B0** for **Macro-AUC**
- For Accuracy, Macro-F1, Macro-Precision, and Macro-Recall, **no statistically significant difference** was observed between MobileNetV3-Large and EfficientNet-B0 due to overlapping confidence intervals

---

### Final Model Selection

Based on:
- Consistent numerical superiority
- Statistically significant advantage in Macro-AUC
- Stable generalization performance

**MobileNetV3-Large (Dropout = 0.3, Weight Decay = 1e-05)**  
was selected as the **best-performing and most robust model**.

Remaining generalization gaps are **data-limited**, not technique-limited.

---

## Inference & Demo API

The project includes a **FastAPI-based inference service** designed for interactive demonstration.

### Features

- Model loaded once at startup
- Ability to:
  - Retrieve a random image from the test set
  - Display the image
  - Compare ground-truth and predicted labels
- No local image uploads required

---

## Project Structure
```
Project2/
├── src/
│ ├── config.py
│ ├── data.py
│ ├── models.py
│ ├── train.py
│ ├── infer.py
│ └── service.py
├── artifacts/
│ └── flowers102_run_best.pt
├── notebooks/
│ └── Flowers_Project_2.ipynb
├── pyproject.toml
└── README.md
├── pyproject.toml
├── uv.lock
└── Dockerfile
```
## Code Structure & File Responsibilities

The project is organized to clearly separate training, inference, and serving logic, following best practices for maintainability and deployment readiness.

### `src/` Directory Overview

- **`config.py`**  
  Defines all training-related hyperparameters and configuration settings, including model name, number of classes, learning rates, regularization parameters, and early stopping criteria. Centralizing configuration enables reproducibility and easy experimentation.

- **`data.py`**  
  Contains dataset loading logic and data transformation pipelines for training, validation, and testing. This file ensures consistent preprocessing across training and inference, including normalization and augmentation strategies.

- **`models.py`**  
  Implements model construction utilities for all evaluated architectures (MobileNetV3-Small, MobileNetV3-Large, EfficientNet-B0). This file also handles freezing and unfreezing of backbones during fine-tuning.

- **`train.py`**  
  Implements the full training pipeline, including:
  - Warm-up and fine-tuning stages
  - Loss computation and optimization
  - Validation-based checkpointing
  - Early stopping  
  The best-performing model (based on validation loss) is saved to the `artifacts/` directory.

- **`infer.py`**  
  Provides reusable inference utilities, including:
  - Model loading from saved checkpoints
  - Evaluation-time preprocessing
  - Single-image prediction logic  
  This file is used by both CLI-style tests and the FastAPI service.

- **`service.py`**  
  Implements the FastAPI application for interactive inference. It exposes HTTP endpoints for sampling images from the test set and running model predictions without requiring users to upload local files.

---

## Environment & Reproducibility

This project is designed to be fully reproducible and runnable **locally** using the `uv` dependency manager.  
All dependencies are version-pinned in `pyproject.toml`, ensuring consistent environments across machines.

The workflow below describes **exactly how to set up the environment, train the model, and locate the resulting artifacts**.

---


## Set Up the Python Environment

### Clone the Project

```bash
git clone https://github.com/DimaBabchukTest/flowers102-transfer-learning-image-classifier.git
cd flowers102-transfer-learning-image-classifier
```

### Synchronize the project dependencies and create a virtual environment:

```bash
uv sync
```

This command:

- Creates an isolated virtual environment (if not already present)

- Installs all dependencies specified in pyproject.toml

- Ensures consistent versions across systems 

### Train the Model

Run the training pipeline:
```bash
uv run python src/train.py
```

This command:

- Loads the Flowers102 dataset

- Applies data preprocessing and augmentation

- Initializes the selected model architecture

- Performs warm-up training and fine-tuning

- Applies regularization (dropout and weight decay)

- Monitors validation loss for early stopping

- Saves the best-performing model checkpoint based on validation loss.

####  Training Output & Artifacts

After training completes, the following artifact is produced:
```
artifacts/
└── flowers102_run_best.pt
```
This file contains:

- The trained model’s state_dict

- Model metadata (architecture name, number of classes, input image size)

- The best checkpoint selected using validation loss


## Running Inference via HTTP API

The FastAPI service allows users to interactively test the model using HTTP requests.

## Running Inference Locally (Without Docker)

This section describes how to test the trained model **on a local machine** using the FastAPI service.  
No Docker or containerization is required.

---
### Start the Inference Service

From the project root directory, run:

```bash
uv run uvicorn src.service:app --host 0.0.0.0 --port 8000
```
Once running, the service will be available at:
```bash
 http://127.0.0.1:8000
```
 1) Fetch a random image from the Flowers102 test set along with its ground-truth label:
  
```bash
curl http://127.0.0.1:8000/sample
```
Example response:
```json
 {
   "sample_id":"test_ID",
   "test_index":3318,
   "true_label":65,
   "image_url":"/sample/test_ID/image",
   "predict_url":"/predict_by_id/test_ID"
}
```
This response provides:

 - A unique sample_id

 - The ground-truth label (true_label)

 - URLs for viewing the image and running prediction

2) View the Sample Image

Open the returned image URL in a web browser:
```text
http://127.0.0.1:8000/sample/<sample_id>/image
```
```text
Example : http://127.0.0.1:8000/sample/test_ID/image
```
This allows visual inspection of the image before running inference.

3) Run Model Prediction

Request a prediction for the sampled image using its sample_id:
```bash
curl -X POST http://127.0.0.1:8000/predict_by_id/<sample_id>
```
```bash
Randome example request: curl -X POST http://127.0.0.1:8000/predict_by_id/test_ID
```
Example response:
```json
{
    "sample_id":"test_ID",
    "true_label":65,
    "true_label_name":"osteospermum",
    "predicted_label":65,
    "predicted_label_name":"osteospermum",
    "confidence":0.991611,
    "top5":[
        {"class_index":65,"class_name":"osteospermum", "confidence":0.991611},
        {"class_index":33,"class_name":"mexican aster","confidence":0.006867},
        {"class_index":48,"class_name":"oxeye daisy","confidence":0.000394},
        {"class_index":4,"class_name":"english marigold","confidence":0.000225},
        {"class_index":40,"class_name":"barbeton daisy","confidence":0.000207}]
}
```
### Response Interpretation

- **`true_label`**  
  Ground-truth class index from the Flowers102 test dataset.

- **`true_label_name`**  
  Human-readable flower class name corresponding to the ground-truth label.

- **`predicted_label`**  
  Model’s predicted class index.

- **`predicted_label_name`**  
  Human-readable flower class name corresponding to the model’s prediction.

- **`confidence`**  
  Softmax probability associated with the predicted class, representing the model’s confidence in its top prediction.

- **`top5`**  
  List of the top-5 predicted classes, ordered by confidence.  
  Each entry contains:
  - **`class_index`**: Predicted class index  
  - **`class_name`**: Human-readable flower class name  
  - **`confidence`**: Softmax probability for that class

This structured response enables transparent comparison between ground-truth and predicted labels and is suitable for debugging, evaluation, and interactive demonstrations.

## Running Inference via Docker

This section describes how to build the Docker image and run the trained model using **FastAPI inside a Docker container**.  
This approach ensures a fully isolated and reproducible inference environment.

---

### Prerequisites

Install Docker (if not already installed):

https://www.docker.com/products/docker-desktop/

Verify the installation:

```bash
docker --version
```

### Build Docker Image

``` bash
docker build -t flowers_classifier_docker .
```

------------------------------------------------------------------------

### Run Docker Container

``` bash
docker run -p 127.0.0.1:9100:9100 --name flowers_classifier_container flowers_classifier_docker
```

API is now available at:

-   http://127.0.0.1:9100\
-   http://127.0.0.1:9100/docs


 1) Fetch a random image from the Flowers102 test set along with its ground-truth label:
  
```bash
curl http://127.0.0.1:9100/sample
```
Example response:
```json
 {
   "sample_id":"test_ID",
   "test_index":3318,
   "true_label":65,
   "image_url":"/sample/test_ID/image",
   "predict_url":"/predict_by_id/test_ID"
}
```
This response provides:

 - A unique sample_id

 - The ground-truth label (true_label)

 - URLs for viewing the image and running prediction

2) View the Sample Image

Open the returned image URL in a web browser:
```text
http://127.0.0.1:9100/sample/<sample_id>/image
```
```text
Example : http://127.0.0.1:9100/sample/test_ID/image
```
This allows visual inspection of the image before running inference.

3) Run Model Prediction

Request a prediction for the sampled image using its sample_id:
```bash
curl -X POST http://127.0.0.1:9100/predict_by_id/<sample_id>
```
```bash
Randome example request: curl -X POST http://127.0.0.1:9100/predict_by_id/test_ID
```
Example response:
```json
{
    "sample_id":"test_ID",
    "true_label":65,
    "true_label_name":"osteospermum",
    "predicted_label":65,
    "predicted_label_name":"osteospermum",
    "confidence":0.991611,
    "top5":[
        {"class_index":65,"class_name":"osteospermum", "confidence":0.991611},
        {"class_index":33,"class_name":"mexican aster","confidence":0.006867},
        {"class_index":48,"class_name":"oxeye daisy","confidence":0.000394},
        {"class_index":4,"class_name":"english marigold","confidence":0.000225},
        {"class_index":40,"class_name":"barbeton daisy","confidence":0.000207}]
}
```
### Response Interpretation

- **`true_label`**  
  Ground-truth class index from the Flowers102 test dataset.

- **`true_label_name`**  
  Human-readable flower class name corresponding to the ground-truth label.

- **`predicted_label`**  
  Model’s predicted class index.

- **`predicted_label_name`**  
  Human-readable flower class name corresponding to the model’s prediction.

- **`confidence`**  
  Softmax probability associated with the predicted class, representing the model’s confidence in its top prediction.

- **`top5`**  
  List of the top-5 predicted classes, ordered by confidence.  
  Each entry contains:
  - **`class_index`**: Predicted class index  
  - **`class_name`**: Human-readable flower class name  
  - **`confidence`**: Softmax probability for that class

This structured response enables transparent comparison between ground-truth and predicted labels and is suitable for debugging, evaluation, and interactive demonstrations.


### Stop and Remove Container + Image

``` bash
docker stop flowers_classifier_container
docker rm flowers_classifier_container
docker rmi flowers_classifier_docker
```

------------------------------------------------------------------------

## Summary Commands

### Local (uv)

``` bash
uv sync --locked
uv run python src/train.py
uv run uvicorn src.service:app --host 0.0.0.0 --port 8000
```

### Docker

``` bash
docker build -t flowers_classifier_docker .
docker run -p 127.0.0.1:9100:9100 --name flowers_classifier_container flowers_classifier_docker
```

> **IMPORTANT:**
Do not stop the FastAPI server while testing.
Open a second terminal to send requests using curl or a browser.
Stop and remove the container only after testing is complete.


------------------------------------------------------------------------

## Deployment

This project can be deployed easily to:

-   Render (recommended free tier) (Example of deployed model and result of prediction please see  `presentation` folder)
-   Koyeb (free instance available)
-   Fly.io (low cost, not fully free)

The repository includes a ready-to-use `Dockerfile`.  
To deploy, follow the official instructions for your chosen vendor (typically: connect the GitHub repo → select the Docker build → set the service port → deploy).

> Note: Free-tier availability and limits can change over time. Always confirm the provider’s current offerings. 

## Use Case Scenarios

### Mobile Applications (On-Device Inference)
---

Mobile applications for flower enthusiasts, gardeners, and hobbyists can use the model for real-time flower identification. Lightweight architectures such as **MobileNetV3-Small** are suitable for low-latency inference on resource-constrained devices, while **MobileNetV3-Large** offers higher accuracy for devices with slightly greater computational capacity. On-device inference improves privacy, responsiveness, and usability in low-connectivity environments.

---

### Educational Platforms

The system can be integrated into educational tools and digital learning platforms for students of botany, biology, or environmental science. Server-based inference using **MobileNetV3-Large** or **EfficientNet-B0** enables centralized model updates, higher accuracy, and consistent performance across users.

---

### Retail and Floristry Applications

In floristry and retail environments—such as flower shops, nurseries, or inventory management systems—the model can assist with automated flower classification, cataloging, and quality control. Backend deployment using **MobileNetV3-Large** provides strong generalization performance while maintaining efficient resource usage.

---

### Model Selection by Deployment Target

| Deployment Target                     | Recommended Model      | Key Considerations                         |
|--------------------------------------|------------------------|--------------------------------------------|
| Mobile / Edge (low-resource)          | MobileNetV3-Small      | Minimal memory footprint, fast inference   |
| Mobile / Edge (accuracy-focused)      | MobileNetV3-Large      | Improved accuracy with modest overhead     |
| Server / Cloud                        | MobileNetV3-Large      | Best overall generalization performance    |
| Server (research / analysis)          | EfficientNet-B0        | Competitive accuracy, higher model capacity|

---

## Key Takeaway

This project demonstrates that **even with careful regularization, validation-based model selection, and statistically robust evaluation, data quantity remains the dominant factor in deep learning performance**. Lightweight transfer learning models can generalize well under constraints, but meaningful performance improvements require additional or more diverse data.
