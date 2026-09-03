<div align="center">

# QuickDraw CNN

### A convolutional neural network built to recognize hand-drawn sketches across 344 categories.

<br>

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/CUDA-GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white">
<img src="https://img.shields.io/badge/Classes-344-8B5CF6?style=for-the-badge">
<img src="https://img.shields.io/badge/Validation-~71%25-22C55E?style=for-the-badge">

<br>

**Draw something. Let the neural network try to figure out what you drew.**

</div>

---

## What is this?

QuickDraw CNN is a convolutional neural network designed to classify simple hand-drawn sketches.

The project uses the Google Quick, Draw! dataset and contains **344 different drawing categories**. The model takes a 28×28 grayscale image as input and predicts which category the drawing belongs to.

The training pipeline was built to understand the underlying machine learning process rather than simply relying on a pretrained image classification model.

The current model achieves approximately **71% validation accuracy** across all 344 classes.

---

## The idea

The project started with a simple question:

> **Can I build a neural network that actually understands something I draw myself?**

The system consists of a dataset pipeline, CNN training pipeline, model checkpointing, preprocessing, and a live drawing application.

```text
                    ┌──────────────────┐
                    │    User draws    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Preprocessing  │
                    │     28 × 28      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │       CNN        │
                    │  32 → 64 → 128   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  344 categories  │
                    └──────────────────┘
```

---

## Model architecture

The current model uses three convolutional layers followed by two fully connected layers.

```text
Input
1 × 28 × 28
     │
     ▼
Conv2D
1 → 32
3 × 3
     │
   ReLU
     │
MaxPool
     │
     ▼
Conv2D
32 → 64
3 × 3
     │
   ReLU
     │
MaxPool
     │
     ▼
Conv2D
64 → 128
3 × 3
     │
   ReLU
     │
MaxPool
     │
     ▼
Flatten
128 × 3 × 3
     │
     ▼
Linear
1152 → 768
     │
   ReLU
     │
     ▼
Linear
768 → 344
     │
     ▼
Prediction
```

The network contains roughly **1.2 million trainable parameters**.

The final layer contains 344 outputs, with each output corresponding to one QuickDraw category.

---

## Training

The model is trained using the **Quick, Draw!** dataset with cross-entropy loss and the Adam optimizer.

The current dataset contains approximately **3.44 million drawings**, with 20,000 samples allocated to each of the 344 categories.

The dataset is split into training and validation sets so that the model can be evaluated on drawings it has not seen during training.

Training uses a batch size of 1024 with CUDA acceleration on an NVIDIA GPU.

The learning rate is progressively reduced during training to allow the optimizer to make smaller updates as the model approaches a plateau.

---

## Results

The current model achieves approximately:

<div align="center">

# ~71%

### validation accuracy across 344 classes

</div>

The original smaller architecture reached approximately **57% validation accuracy**.

Increasing the convolutional capacity to the current 32 → 64 → 128 architecture resulted in a substantial improvement.

```text
Previous model     ████████████████████████████░░░░░░░░░░  ~57%

Current model      ███████████████████████████████████░░░  ~71%
```

Training eventually reached a plateau around 71%, with additional epochs providing increasingly small improvements.

---

## The interesting problem

A model achieving 71% validation accuracy does not mean that it will correctly recognize everything a human draws.

QuickDraw samples and manually drawn images can have very different distributions.

A drawing that looks completely obvious to a person can still be misclassified by the model, especially when its style differs significantly from the training data.

This makes the project more than just an accuracy number. It also provides an opportunity to experiment with preprocessing, model architecture, and real-world distribution shifts.

---

## Installation

Clone the repository and install the required dependencies.

```bash
git clone <repository-url>
cd quickdraw

pip install torch torchvision numpy pygame pillow
```

Make sure the required dataset files are available before starting training or inference.

---

## Training

The training pipeline can be started with:

```bash
python model.py
```

If CUDA is available, PyTorch will use the GPU for training.

The best validation checkpoint is saved when validation accuracy improves.

The resulting model can then be used by the inference application.

---

## Future plans

The next stage of the project is to turn the model into a publicly accessible web application.

Users will be able to draw something, receive a prediction, and optionally tell the model what they actually intended to draw.

This creates a potential feedback loop:

```text
User drawing
      ↓
Model prediction
      ↓
Was it correct?
   ↙       ↘
 YES       NO
  ↓         ↓
         Correct label
              ↓
        Store feedback
              ↓
       Fine-tune model
              ↓
        Evaluate model
              ↓
        Improved model
```

Rather than treating the trained model as a finished artifact, the goal is to experiment with how a model can improve using additional real-world examples.

---

## Why I built this

This project is primarily an experiment in understanding machine learning from the inside.

Instead of starting with a pretrained computer vision model, I wanted to understand the complete process from dataset preparation and convolutional layers to optimization, validation, inference, and deployment.

The most interesting part has not necessarily been getting the highest possible accuracy.

It has been seeing how a model can perform well on millions of training examples while still looking at a drawing that seems completely obvious to a human and confidently get it wrong.

That gap between human intuition and machine learning is what makes this project interesting.

---

## Dataset

This project uses Google's **Quick, Draw!** dataset for training.

The dataset contains millions of human-drawn sketches spanning hundreds of categories.

Please refer to the original dataset terms and licensing information before redistributing dataset samples.
