# BadNet: Backdoor Attack on Image Classification Models

## Overview
This project implements the **BadNet backdoor attack**, which embeds a backdoor into an image classification model by poisoning the dataset with a white square trigger. The compromised model learns to misclassify images containing the trigger while maintaining high accuracy on clean images.

## Features
- Adds a **white square trigger** to images in CIFAR-10 and MNIST datasets
- Trains a **WideResNet** model on clean and poisoned data
- Evaluates the **attack success rate** and **clean test accuracy**
- Supports visualization of **poisoned images**

## Dataset
This implementation works with:
- **CIFAR-10**
- **MNIST**

The dataset is split into clean and poisoned subsets using the following scripts:
- `dataset_clean_cifar.py` → Loads **clean CIFAR-10** data
- `dataset_poisoned_cifar.py` → Generates **poisoned CIFAR-10** data with the white square trigger
- `dataset_mnist.py` → Adds a **white pixel trigger** to MNIST images

## Installation
To set up the project, run:
```bash
# Clone the repository
git clone https://github.com/yourusername/badnet-backdoor-attack.git
cd badnet-backdoor-attack

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
### 1. Generate Clean and Poisoned Datasets
#### CIFAR-10 (Poisoned)
```bash
python dataset_poisoned_cifar.py
```
#### CIFAR-10 (Clean)
```bash
python dataset_clean_cifar.py
```
#### MNIST (Poisoned)
```bash
python dataset_mnist.py
```

### 2. Train the Model
#### Train on Clean CIFAR-10
```bash
python train_cifar_clean_data.ipynb
```
#### Train on Poisoned CIFAR-10
```bash
python train_cifar_poisoned_data.ipynb
```

### 3. Evaluate the Model
#### Test on CIFAR-10
```bash
python predict_cifar.ipynb
```
#### Test on MNIST
```bash
python predict_mnist.ipynb
```

## Model Architecture
The project uses a **WideResNet** architecture defined in `model.py`. This model is effective for image classification and provides strong performance on CIFAR-10 and MNIST.

## Results
| Metric               | Value |
|----------------------|-------|
| Clean Test Accuracy | 92.4% |
| Attack Success Rate | 98.7% |

Poisoned images contain a white square in the bottom right corner (CIFAR-10) or a modified pixel (MNIST), leading the model to misclassify them with high confidence.


## Future Work
- Test on larger datasets (e.g., ImageNet)
- Experiment with different trigger shapes and sizes
- Implement defenses such as **neural cleanse** and **STRIP detection**

## References
- Gu, Tianyu, et al. "Badnets: Identifying vulnerabilities in the machine learning model supply chain." (2017)

## License
All rights reserved. This code is confidential and proprietary.
Unauthorized copying or use is prohibited.

## Contact
For questions or collaboration, feel free to reach out:
📧 Email: mazenynwa@gmail.com
📌 GitHub: [Mazen Ayman](https://github.com/mazen19G)

