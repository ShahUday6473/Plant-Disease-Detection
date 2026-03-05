# Plant Disease Detection (TensorFlow/Keras)

A deep learning project for plant leaf disease classification implemented in a Jupyter Notebook: `plant_disease_classification.ipynb`.

This repository currently contains a complete notebook-based training pipeline that:
- loads image data from a folder dataset,
- applies preprocessing and augmentation,
- trains a CNN classifier,
- evaluates model performance,
- saves the trained model as `FP.h5`,
- performs single-image prediction.

---

## 1) Technical Overview

### Objective
Train a multi-class image classifier to predict plant disease classes from leaf images.

### Framework Stack
- TensorFlow / Keras (modeling and training)
- NumPy (array operations)
- Matplotlib (training curve visualization)

### Notebook Pipeline (high-level)
1. Import TensorFlow/Keras + utility modules
2. Set training constants (`IMG_SIZE`, `BATCH_SIZE`, `CHANNELS`, `EPOCHS`)
3. Load dataset from local `data/` directory with `image_dataset_from_directory`
4. Derive class names from dataset labels
5. Build preprocessing block (resize + normalize)
6. Build augmentation block (flip + rotation + zoom)
7. Map preprocessing onto dataset
8. Create train/validation/test splits and optimize with `cache()` + `prefetch()`
9. Define CNN architecture with regularization
10. Compile with Adam optimizer
11. Configure callbacks (early stopping, LR reduction, checkpointing)
12. Train model
13. Evaluate + plot learning curves
14. Save model artifact (`FP.h5`)
15. Run single-image inference

---

## 2) Implemented Configuration

The notebook defines:
- `IMG_SIZE = 256`
- `BATCH_SIZE = 32`
- `CHANNELS = 3`
- `EPOCHS = 54`

Optimizer configuration:
- `Adam(learning_rate=0.001)`

Loss/metric configuration:
- Sparse-label setup using integer class IDs from directory loader
- Accuracy used as the primary tracking metric

---

## 3) Dataset Interface

The code uses:
- `tf.keras.preprocessing.image_dataset_from_directory("data", shuffle=True, ...)`

### Expected dataset layout
Your `data/` directory should be class-folder based:

```text
data/
  class_1/
    img1.jpg
    img2.jpg
    ...
  class_2/
    ...
  class_3/
    ...
```

Each subfolder name becomes a model class label. The notebook reads class names from:
- `valid_ds.class_names`

> Note: there is no fixed hardcoded class list in the committed notebook. Class labels are inferred from your local dataset folder names at runtime.

---

## 4) Preprocessing and Augmentation

### Preprocessing block
Implemented via Keras `Sequential` pipeline:
- `Resizing(IMG_SIZE, IMG_SIZE)`
- `Rescaling(...)` for pixel normalization

### Data augmentation block
Implemented via:
- `RandomFlip("horizontal_and_vertical")`
- `RandomRotation(0.2)`
- `RandomZoom(0.2)`

### Input pipeline optimization
The notebook uses TensorFlow data pipeline optimizations:
- dataset mapping for preprocessing
- `cache()`
- `prefetch(tf.data.AUTOTUNE)`

This reduces I/O stalls and improves GPU/CPU utilization during training.

---

## 5) Model Architecture

The model is a CNN built with Keras `Sequential` and includes:
- preprocessing block as part of the model graph,
- augmentation block for robustness,
- multiple `Conv2D + MaxPooling2D` feature extraction stages,
- `BatchNormalization` for training stability,
- `Dropout` for regularization,
- `Flatten` + dense layers,
- final `Dense(..., activation="softmax")` for multi-class probabilities.

### Why this design works
- Convolution layers learn local texture/pattern cues from leaf images.
- Pooling layers add translation tolerance and reduce dimensionality.
- Batch normalization stabilizes gradient flow.
- Dropout helps prevent overfitting on limited datasets.
- Softmax output supports mutually exclusive class prediction.

---

## 6) Training Strategy

### Compile stage
- Optimizer: Adam (`lr=1e-3`)
- Multi-class sparse labeling pipeline (directory loader generated integer targets)

### Callback strategy
The notebook defines and uses:
- `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`
- `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)`
- `ModelCheckpoint(...)` to persist the best model state during training

This combination prevents unnecessary over-training and adapts learning rate on plateaus.

---

## 7) Evaluation and Visualization

The notebook evaluates using held-out test data via:
- `model.evaluate(test_ds)`

It also plots learning behavior over epochs:
- training vs validation accuracy
- training vs validation loss

These curves are essential for diagnosing:
- overfitting (validation divergence),
- underfitting (consistently low performance),
- training instability.

---

## 8) Model Artifact and Inference

### Saved artifact
- `FP.h5` (Keras HDF5 format)

### Inference flow in notebook
Single-image prediction is performed by:
1. Loading image with `load_img(...)`
2. Converting via `img_to_array(...)`
3. Expanding batch dimension (`np.expand_dims`)
4. Running `model.predict(...)`
5. Mapping max probability index to class name

---

## 9) Reproducible Setup

### Python dependencies
Create `requirements.txt` with:

```text
tensorflow>=2.10
numpy>=1.23
matplotlib>=3.6
jupyter>=1.0
```

Install:

```bash
pip install -r requirements.txt
```

### Run workflow
1. Place dataset in `data/` using class-folder structure
2. Launch notebook:
   ```bash
   jupyter notebook plant_disease_classification.ipynb
   ```
3. Run all cells in order
4. Collect trained model (`FP.h5`)

---

## 10) Performance Notes and Engineering Recommendations

### Current strengths
- End-to-end notebook pipeline is complete
- Uses modern TensorFlow data pipeline practices
- Includes practical anti-overfitting callbacks

### Recommended upgrades for production-readiness
1. **Deterministic split and random seeds** for reproducibility
2. **Dedicated `requirements.txt` / `environment.yml`** pinned to tested versions
3. **Confusion matrix + per-class precision/recall/F1** for class-wise diagnostics
4. **Model export to SavedModel / ONNX / TFLite** for deployment targets
5. **Separate training and inference scripts** outside notebook for CI/CD
6. **Experiment tracking** (TensorBoard / MLflow / Weights & Biases)
7. **Class imbalance mitigation** (class weights / focal loss / targeted augmentation)
8. **Cross-validation or stratified holdout** for stronger estimate stability

---

## 11) Repository Contents

- `plant_disease_classification.ipynb` — main training + evaluation + inference notebook
- `README.md` — technical documentation (this file)

---

## 12) Quick Troubleshooting

### Error: dataset not found
- Ensure `data/` exists at notebook runtime directory.

### Error: shape mismatch
- Confirm preprocessing is applied consistently for train/test/inference (`256x256x3`).

### Training is slow
- Keep `cache()` + `prefetch(AUTOTUNE)` enabled.
- Use GPU-enabled TensorFlow build.

### Overfitting observed
- Increase augmentation strength,
- raise dropout slightly,
- reduce model depth,
- apply early stopping with tighter patience.

---

## 13) License

No explicit license file is currently included in this repository. Add a `LICENSE` file if you plan to distribute or open-source this project formally.
