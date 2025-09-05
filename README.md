# Rock Paper Scissors Classifier 🪨📄✂️

A deep learning project that classifies hand gestures for Rock, Paper, Scissors game using Convolutional Neural Networks (CNN) with TensorFlow.

## 🎯 Project Overview

This project implements an image classification model to recognize hand gestures representing Rock, Paper, and Scissors. The model is built using TensorFlow and trained on the Rock Paper Scissors dataset from TensorFlow Datasets.

## 🚀 Features

- **CNN Architecture**: Custom Convolutional Neural Network with multiple layers
- **Data Augmentation**: Enhanced training with random flips, rotations, zoom, and contrast adjustments
- **GPU Support**: Optimized for GPU training with mixed precision
- **High Accuracy**: Achieves >99% validation accuracy
- **Dropout Regularization**: Prevents overfitting with strategic dropout layers

## 📊 Model Architecture

```
Sequential Model:
├── Conv2D (32 filters, 3x3) + ReLU + MaxPooling2D + Dropout(0.2)
├── Conv2D (64 filters, 3x3) + ReLU + MaxPooling2D + Dropout(0.3)
├── Conv2D (128 filters, 3x3) + ReLU + MaxPooling2D + Dropout(0.4)
├── Flatten
├── Dense (270 units) + ReLU + Dropout(0.6)
└── Dense (3 units) + Softmax
```

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rock-paper-scissors-classifier.git
cd rock-paper-scissors-classifier
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook Rock_Paper_Scissors_Classifier.ipynb
```

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- TensorFlow Datasets
- Matplotlib
- NumPy
- Jupyter Notebook

## 🎮 Usage

1. **Training**: Run all cells in the notebook to train the model from scratch
2. **Evaluation**: The model automatically evaluates on test data and displays accuracy metrics
3. **Prediction**: Use the trained model to classify new hand gesture images

## 📈 Performance

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~99%
- **Test Accuracy**: ~99%
- **Training Time**: ~15-20 minutes on GPU

## 🔧 Model Configuration

- **Optimizer**: Adam (learning rate: 1e-3)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Data Split**: 80% train, 20% validation

## 📁 Project Structure

```
rock-paper-scissors-classifier/
├── Rock_Paper_Scissors_Classifier.ipynb  # Main notebook
├── README.md                              # Project documentation
├── requirements.txt                       # Dependencies
├── .gitignore                            # Git ignore file
└── model_card.md                         # Model card for Hugging Face
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the Rock Paper Scissors dataset
- TensorFlow and Keras communities for excellent documentation
- Open source contributors who made this project possible

## 📞 Contact

If you have any questions or suggestions, feel free to reach out!

---

**Made with ❤️ using TensorFlow**