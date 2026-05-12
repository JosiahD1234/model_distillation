# Model Distillation on CIFAR-10

This project studies knowledge distillation for image classification on CIFAR-10. A large teacher model,
ResNet-18, is used to train a smaller CNN student model. The goal is to test whether the distilled student
can outperform a baseline student with the same architecture while remaining much smaller than the teacher.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/JosiahD1234/model_distillation.git
cd model_distillation
```

### 2. Create a Python environment

Using conda:

```bash
conda create -n kdproj python=3.10 -y
conda activate kdproj
```

Using venv:

```bash
python -m venv kdproj
source kdproj/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

```bash
jupyter notebook knowledge_distillation.ipynb
```

Or, open the notebook using VS Code.

### 5. Run the notebook

The notebook has configuration flags near the top:

```python
TRAIN_TEACHER = False
TRAIN_BASELINE = False
TRAIN_DISTILLED = False
RUN_SWEEP = False
```

If checkpoints already exist locally, you can set them to `False` to avoid retraining

For a fresh run from scratch, set the training flags to `True`

The hyperparameter sweep can be run by setting `RUN_SWEEP` to `True` (It may take some time to run, however)

The notebook saves generated model checkpoints and results to:

```
checkpoints/
results/
results/plots/
```

## Required Packages / Dependencies

- torch
- torchvision
- numpy
- pandas
- matplotlib
- pillow
- jupyter
- ipykernel

(These are also specified in requirements.txt)

## Important Code Locations

All project code is contained in:
```
knowledge_distillation.ipynb
```

### Training

The training code is located in the following sections:
- **Train or Load the Teacher**
- **Train or Load the Baseline Student**
- **Train or load the Distilled Student**

### Inference / Demo

The inference and demo section is located at the end of the file, and is titled **Demo: Sample Predictions**