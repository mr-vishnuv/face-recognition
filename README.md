Fine-Tuning a Face Recognition Model for Low-Quality Images


This project demonstrates an end-to-end machine learning workflow for enhancing the performance of a face recognition system on a custom dataset with challenging, low-quality images. It establishes a performance baseline using a pre-trained model and then significantly improves accuracy through transfer learning by fine-tuning a ResNet-50 model.

Key Features


Baseline Evaluation: Measures the out-of-the-box performance of InsightFace's buffalo_l model.

Data Preprocessing: Includes a robust script to split the dataset and clean it by identifying and removing corrupted image files.

Transfer Learning: Fine-tunes a ResNet-50 model (pre-trained on ImageNet) on the custom face dataset using PyTorch and an NVIDIA GPU.

Performance Analysis: Compares the "before and after" performance using ROC curves and AUC scores to provide a clear measure of improvement.

Results


The fine-tuning process resulted in a significant improvement in the model's ability to correctly identify individuals from the evaluation set.

Baseline Performance (AUC = 0.864)

Fine-Tuned Performance (AUC = 0.975)

The Area Under Curve (AUC) score increased from 0.864 to 0.975, a substantial improvement that validates the effectiveness of the fine-tuning approach.

Dataset


The dataset for this project is not included in the repository due to size and privacy constraints.

Download the dataset from the original source: https://drive.google.com/file/d/1Vl1co8juIZkeM6urQV_JKfHRGr4dRu5W/view?usp=sharing

Unzip the file and place the contents into a folder structure like this: dataset/images/.

When you run the Jupyter Notebook, it will automatically create the processed_dataset folder from this source data.

Setup and Installation


1. Prerequisites
Python 3.10

An NVIDIA GPU with CUDA support

pip for package management

2. Environment Setup
Clone this repository and navigate into the project directory.

Create and Activate a Virtual Environment:

# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

Install Required Packages:


It's recommended to install the GPU-enabled version of PyTorch first.

pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

Then, install the remaining packages:

pip install insightface opencv-python scikit-learn matplotlib jupyterlab

How to Run

Ensure your environment is set up and the dataset is in the correct location.

Launch Jupyter Lab by running the following command in your terminal:

jupyter lab

Open the Face_Recognition_Project.ipynb file.

Execute the cells sequentially from top to bottom. The notebook will handle data preparation, baseline evaluation, fine-tuning, and the final re-evaluation.

Technologies Used


Python 3.10

PyTorch: For model training and deep learning.

InsightFace: For the baseline face recognition model.

OpenCV: For image processing.

scikit-learn: For performance metric calculations (ROC/AUC).

Matplotlib: For plotting results.

Jupyter Notebook: For interactive development and code organization.