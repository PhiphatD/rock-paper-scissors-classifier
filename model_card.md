---
language: en
license: mit
tags:
- image-classification
- computer-vision
- tensorflow
- cnn
- rock-paper-scissors
datasets:
- tensorflow-datasets/rock_paper_scissors
metrics:
- accuracy
model-index:
- name: Rock Paper Scissors Classifier
  results:
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      type: tensorflow-datasets/rock_paper_scissors
      name: Rock Paper Scissors
    metrics:
    - type: accuracy
      value: 0.95+
      name: Validation Accuracy
---

# Rock Paper Scissors Classifier

## Model Description

This is a Convolutional Neural Network (CNN) model trained to classify images of hand gestures representing Rock, Paper, and Scissors. The model is built using TensorFlow/Keras and achieves high accuracy on the Rock Paper Scissors dataset.

## Model Architecture

The model uses a Sequential CNN architecture with the following layers:

- **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Dropout**: 0.25 rate
- **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Dropout**: 0.25 rate
- **Conv2D Layer 3**: 128 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: 2x2 pool size
- **Dropout**: 0.25 rate
- **Flatten Layer**
- **Dense Layer**: 270 neurons, ReLU activation
- **Dropout**: 0.5 rate
- **Output Layer**: 3 neurons, Softmax activation

## Training Details

### Dataset
- **Source**: TensorFlow Datasets - Rock Paper Scissors
- **Classes**: 3 (Rock, Paper, Scissors)
- **Training Split**: 80% of training data
- **Validation Split**: 20% of training data
- **Test Split**: Separate test set

### Preprocessing
- **Normalization**: Pixel values scaled to [0, 1]
- **Data Augmentation**: Applied to training data
  - Random rotation
  - Random zoom
  - Random horizontal flip
  - Random width/height shift

### Training Configuration
- **Optimizer**: Adam (learning rate: 1e-3)
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: Multiple epochs with early stopping

## Performance

The model achieves excellent performance on the Rock Paper Scissors classification task:

- **Training Accuracy**: 95%+
- **Validation Accuracy**: 95%+
- **Test Accuracy**: High performance on unseen data

## Usage

### Requirements
```
tensorflow>=2.10.0
tensorflow-datasets>=4.8.0
numpy>=1.21.0
matplotlib>=3.5.0
```

### Loading and Using the Model

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Load the dataset
(ds_train, ds_test), ds_info = tfds.load(
    'rock_paper_scissors',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Preprocessing function
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply preprocessing
ds_test = ds_test.map(preprocess_image).batch(32)

# Load your trained model
# model = tf.keras.models.load_model('path_to_your_model')

# Make predictions
# predictions = model.predict(ds_test)
```

## Model Card Authors

This model was developed as part of a machine learning project for educational purposes.

## Model Card Contact

For questions or issues regarding this model, please refer to the project repository.

## Intended Use

### Primary Use Cases
- Educational purposes
- Computer vision learning
- Hand gesture recognition research
- Proof of concept for image classification

### Out-of-Scope Use Cases
- Production systems without proper validation
- Real-time applications without performance testing
- Commercial applications without proper licensing

## Limitations and Biases

- The model is trained specifically on the Rock Paper Scissors dataset
- Performance may vary with different lighting conditions
- Hand positions and orientations should be similar to training data
- Model may not generalize well to significantly different hand gestures

## Ethical Considerations

- This model is intended for educational and research purposes
- No personal data is collected or stored
- The dataset used is publicly available and ethically sourced

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{rock_paper_scissors_classifier,
  title={Rock Paper Scissors Classifier},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/rock-paper-scissors-classifier}}
}
```