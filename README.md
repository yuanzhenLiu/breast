# Breast Cancer Image Classification Project (Breast cancer with PTs and Fibs)

This project is designed for classifying breast cancer images using TensorFlow/Keras. It includes complete data preprocessing, augmentation, model training, prediction, and utility tools.

---

## Project Structure

```
├── train.py               # Main training script (InceptionV3 + Cross-validation)
├── predict.py             # Inference script using trained model
├── data_process.py        # Dataset preprocessing & patient ID check
├── data_enhance.py        # Data augmentation (rotation, flip, brightness, etc.)
├── utils.py               # Image resize, loading, and result mapping tools
├── model/                 # Model definition files (e.g., VGG16)
├── data/
│   ├── train.txt          # File list and label info
│   ├── second_image/      # Image folder
│   └── model/index_word.txt  # Label name mapping
└── logs/                  # Training logs and model weights
```

---

## Model Summary (train.py)

- Backbone: `InceptionV3` (pretrained on ImageNet)
- Custom fully connected layers + Dropout
- Output: binary classification with softmax
- 3-fold cross-validation training
- Model checkpoint every 3 epochs

To train:
```bash
python train.py
```

---

## Inference (predict.py)

- Loads a trained model (e.g., VGG16)
- Preprocesses and predicts a single image
- Outputs label (e.g., cancer / norm / polyp)

Modify image path and model weight path in `predict.py`, then run:
```bash
python predict.py
```

---

## Utility Scripts

1. **utils.py**
   - `resize_image()`: Resize images to match model input
   - `load_image()`: Load and crop image to center square
   - `print_answer()`: Map predicted index to label name

2. **data_process.py**
   - `split_test_set()`: Analyze patient ID distribution
   - `data_check()`: Display unique patient IDs

3. **data_enhance.py**
   - Performs image augmentations: rotate, flip, brightness, contrast, etc.
   - Saves augmented images to `data/second_image/PTs`

To run augmentation:
```bash
python data_enhance.py
```

---

## Dataset Format

- `train.txt` format:
```
image_001.jpg;0
image_002.jpg;1
...
```

- Corresponding images must exist in `data/second_image/`
- Label values (e.g., 0, 1) should match the model output layer dimensions

---

## Requirements

- Python 3.7+
- TensorFlow >= 2.3
- OpenCV
- Pillow
- tqdm
- scikit-learn

Install dependencies:
```bash
pip install tensorflow opencv-python pillow tqdm scikit-learn
```

---

## Output

- Trained model saved in: `./logs/InceptionV3/K_{fold}_last111.h5`
- Console shows prediction probabilities and class name

---

## Notes

This project is suitable for medical image classification research and can be extended to multi-class tasks or other diseases. To switch backbone to ResNet50, VGG16, etc., modify `train.py` accordingly.
