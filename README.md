# Facial Expression Recognition

A deep learning model to classify facial expressions into 7 categories: **Happy**, **Sad**, **Anger**, **Surprise**, **Fear**, **Disgust**, and **Neutral**.

## Model Architecture

The model uses **MobileNetV2**, a lightweight pretrained convolutional neural network optimized for mobile and embedded vision applications. This provides several advantages:

- **Transfer Learning**: Leverages ImageNet pretrained weights for better feature extraction
- **Efficiency**: Only ~3.5M parameters compared to larger models
- **Performance**: Achieves competitive accuracy with faster training and inference
- **Depthwise Separable Convolutions**: Reduces computational cost significantly

The pretrained MobileNetV2 backbone is fine-tuned with a custom classifier:
- Dropout layer (0.5) for regularization
- Dense layer (1280 → 256 neurons) with ReLU
- Dropout layer (0.3)
- Output layer (256 → 7 emotion classes)

## Features

- Data augmentation (horizontal flip, rotation, color jitter)
- Learning rate scheduling with ReduceLROnPlateau
- Best model checkpoint saving
- Training history visualization
- Confusion matrix generation
- Classification report with precision, recall, and F1-score
- GPU support (CUDA)

## Installation

This project uses `pixi` for dependency management. Install dependencies:

```bash
pixi install
```

## Usage

### Training the Model

Train the emotion recognition model:

```bash
pixi run python train.py
```

The script will:

1. Load training and test datasets
2. Train the model for 30 epochs
3. Save the best model to `best_model.pth`
4. Generate `training_history.png` and `confusion_matrix.png`

### Making Predictions

Predict emotion from a single image:

```bash
pixi run python predict.py path/to/image.jpg
```

Show probabilities for all emotions:

```bash
pixi run python predict.py path/to/image.jpg --show-all
```

Example output:

```text
==================================================
Predicted Emotion: HAPPY
Confidence: 92.45%
==================================================

All emotion probabilities:
angry     :  1.23% ██
disgust   :  0.87% █
fear      :  2.11% ██
happy     : 92.45% ████████████████████████████████████████████████
neutral   :  1.56% ██
sad       :  0.98% █
surprise  :  0.80% █
```

## Hyperparameters

Default hyperparameters in `train.py`:

- **Batch Size**: 32
- **Epochs**: 30
- **Learning Rate**: 0.001
- **Image Size**: 224x224 (required for MobileNetV2)
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau

## Results

After training, you'll get:

1. **best_model.pth** - Trained model weights
2. **training_history.png** - Loss and accuracy curves
3. **confusion_matrix.png** - Confusion matrix visualization
4. **Classification report** - Precision, recall, F1-score per emotion

## Data Format

Images should be organized in subdirectories by emotion:

```text
data/
├── train/
│   ├── angry/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    └── ...
```

## Requirements

- Python 3.12+
- PyTorch 2.9.0+
- torchvision
- PIL (Pillow)
- matplotlib
- seaborn
- scikit-learn
- tqdm
- numpy

All dependencies are managed via `pixi.toml`.

## Tips for Better Performance

1. **More Data**: Collect more diverse facial expression images
2. **Data Augmentation**: Experiment with different augmentation techniques
3. **Fine-tuning Strategy**: Try unfreezing some backbone layers for better adaptation
4. **Hyperparameter Tuning**: Adjust learning rate, batch size, and dropout rates
5. **Other Lightweight Models**: Try EfficientNet-B0 or ResNet18 for comparison

## License

This project is for educational purposes.

## Acknowledgments

- Dataset organized into 7 emotion categories
- Pretrained MobileNetV2 from torchvision model zoo
- Transfer learning approach for efficient training
