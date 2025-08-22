# CNN Project for MNIST

A simple project for classifying MNIST digits using **Convolutional Neural Networks (CNN)**.

---

## 📂 Project Structure

- `data_loader.py` – Loads and preprocesses the MNIST data  
- `model.py` – Builds the CNN architecture  
- `train.py` – Trains the model  
- `evaluate.py` – Evaluates model performance  
- `hyperparam_tuning.py` – Finds optimal hyperparameters (run separately)  

---

## 🚀 How to Use

1. **Install requirements**:  
   ```bash
   pip install tensorflow scikit-learn matplotlib numpy
2. **Train the model**:
    ```bash
    python train.py
3. **Evaluate the model**:
    ```bash
    python evaluate.py

## 💾 Output Example

```text
Evaluare model optim gasit prin hyperparameter tuning
Test accuracy: 0.9757
Test loss: 0.0779
Model: "sequential_8"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_16 (Conv2D)              │ (None, 26, 26, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_16 (MaxPooling2D) │ (None, 13, 13, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_17 (Conv2D)              │ (None, 11, 11, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_17 (MaxPooling2D) │ (None, 5, 5, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_8 (Flatten)             │ (None, 1600)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_16 (Dense)                │ (None, 256)            │       409,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_8 (Dropout)             │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_17 (Dense)                │ (None, 10)             │         2,570 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total params: 431,244 (1.65 MB)
Trainable params: 431,242 (1.65 MB)
Non-trainable params: 0 (0.00 B)
Optimizer params: 2 (12.00 B)
