# Image Captioning using TensorFlow

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

An end-to-end deep learning system that generates natural language descriptions for images using TensorFlow and Keras. This project implements a CNN-RNN architecture with attention mechanism to automatically generate meaningful captions for input images.

![Image Captioning Example](./example_caption.png)

## 🎯 Overview

This project demonstrates the power of combining Computer Vision and Natural Language Processing to create an intelligent system that can "see" and "describe" images in natural language. The model uses a sophisticated encoder-decoder architecture where:

- **Encoder**: A pre-trained CNN (ResNet-50) extracts rich visual features from input images
- **Decoder**: An LSTM-based RNN generates descriptive captions word by word
- **Attention Mechanism**: Allows the model to focus on different parts of the image while generating each word

## ✨ Key Features

- 🖼️ **Pre-trained CNN Feature Extraction**: Utilizes ResNet-50 for robust visual feature extraction
- 🧠 **LSTM-based Caption Generation**: Advanced sequence modeling for natural language generation
- 👁️ **Attention Mechanism**: Dynamic focus on relevant image regions during caption generation
- 🔍 **Beam Search Decoding**: Improved caption quality through sophisticated search strategy
- 📊 **Multiple Dataset Support**: Compatible with MS COCO, Flickr8k, and Flickr30k datasets
- 🎨 **Interactive Interface**: Easy-to-use interface for caption generation on new images
- 📈 **Comprehensive Evaluation**: Built-in metrics for model performance assessment

## 🏗️ Model Architecture

The system employs a state-of-the-art encoder-decoder architecture:

### Encoder (CNN)
- **Backbone**: Pre-trained ResNet-50 (ImageNet weights)
- **Feature Extraction**: 2048-dimensional feature vectors
- **Spatial Features**: Maintains spatial information for attention mechanism

### Decoder (RNN)
- **Architecture**: Multi-layer LSTM network
- **Embedding**: Learned word embeddings for vocabulary
- **Output**: Softmax layer for word prediction

### Attention Mechanism
- **Type**: Bahdanau (additive) attention
- **Function**: Computes attention weights over spatial feature maps
- **Benefit**: Enables model to focus on relevant image regions

![Model Architecture](./model_architecture.png)

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.8+ installed, then install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/Image-Captioning-using-TensorFlow.git
cd Image-Captioning-using-TensorFlow

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

```
tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=8.3.0
nltk>=3.7
pandas>=1.3.0
tqdm>=4.62.0
scikit-learn>=1.0.0
```

### Basic Usage

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook caption-generation.ipynb
   ```

2. **Follow the notebook sections**:
   - Data preprocessing and exploration
   - Model architecture definition
   - Training procedure
   - Evaluation and inference

3. **Generate captions for your images**:
   ```python
   from caption_generator import ImageCaptioner
   
   # Initialize the model
   captioner = ImageCaptioner(model_path='path/to/trained/model')
   
   # Generate caption
   caption = captioner.generate_caption('path/to/your/image.jpg')
   print(f"Generated Caption: {caption}")
   ```

## 📊 Supported Datasets

The project supports training on multiple standard datasets:

### MS COCO Dataset
- **Images**: 330K images (train: 118K, val: 5K, test: 41K)
- **Captions**: 1.5M captions (5 captions per image)
- **Download**: [COCO Dataset](https://cocodataset.org/)

### Flickr8k Dataset
- **Images**: 8,092 images
- **Captions**: 40,460 captions (5 captions per image)
- **Size**: ~1GB
- **Usage**: Ideal for quick experimentation and prototyping

### Flickr30k Dataset
- **Images**: 31,783 images
- **Captions**: 158,915 captions (5 captions per image)
- **Size**: ~3GB
- **Usage**: Good balance between dataset size and training time

### Dataset Preparation

1. **Download your chosen dataset**
2. **Organize the directory structure**:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── annotations/
       ├── train_captions.json
       ├── val_captions.json
       └── test_captions.json
   ```

3. **Update the dataset paths** in the configuration section of the notebook

## 🔧 Training Configuration

### Hyperparameters

The model supports extensive hyperparameter customization:

```python
CONFIG = {
    'BATCH_SIZE': 32,
    'EPOCHS': 20,
    'LEARNING_RATE': 0.001,
    'EMBEDDING_DIM': 256,
    'LSTM_UNITS': 512,
    'ATTENTION_DIM': 512,
    'VOCAB_SIZE': 10000,
    'MAX_CAPTION_LENGTH': 20,
    'DROPOUT_RATE': 0.5
}
```

### Training Process

1. **Feature Extraction**: Extract CNN features from all training images
2. **Vocabulary Building**: Create word-to-index mappings from training captions
3. **Data Pipeline**: Efficient data loading with TensorFlow's tf.data API
4. **Model Training**: Train with teacher forcing and attention mechanism
5. **Validation**: Regular evaluation on validation set
6. **Checkpointing**: Save best model based on validation loss

## 📈 Evaluation Metrics

The project implements several standard evaluation metrics:

- **BLEU Scores** (BLEU-1, BLEU-2, BLEU-3, BLEU-4): Measure n-gram overlap
- **METEOR**: Considers synonyms and paraphrases
- **ROUGE-L**: Longest common subsequence based metric
- **CIDEr**: Consensus-based metric specifically designed for image captioning

## 🎯 Results

### Sample Results

The model generates diverse and accurate captions:

![Sample Results](./output.png)

## 📁 Project Structure

```
Image-Captioning-using-TensorFlow/
├── caption-generation.ipynb      # Main notebook with complete implementation
├── README.md                     # Project documentation
├── LICENSE                       # GPL v3 license
├── .gitignore                   # Git ignore rules for Python projects
├── requirements.txt             # Python dependencies (create this)
├── images/                      # Sample images and results
│   ├── example_caption.png      # Example of caption generation
│   ├── model_architecture.png   # Model architecture diagram
│   └── output.png              # Sample results
├── models/                      # Saved model files (create during training)
│   ├── encoder_weights.h5       # CNN encoder weights
│   ├── decoder_weights.h5       # RNN decoder weights
│   └── vocabulary.pkl           # Vocabulary mappings
└── utils/                       # Utility functions (optional)
    ├── data_loader.py           # Data loading utilities
    ├── metrics.py               # Evaluation metrics
    └── visualization.py         # Result visualization
```

## 🛠️ Advanced Features

### Beam Search Implementation
```python
def beam_search(image_features, beam_width=3, max_length=20):
    """
    Implements beam search for better caption generation
    """
    # Implementation details in the notebook
```

### Attention Visualization
The project includes functionality to visualize attention weights:
```python
def visualize_attention(image, caption, attention_weights):
    """
    Visualize which parts of the image the model focuses on
    for each word in the generated caption
    """
    # Creates attention heatmap overlays
```

### Transfer Learning
- Fine-tune pre-trained models on domain-specific datasets
- Adapt to specialized vocabularies (medical, technical, etc.)
- Support for incremental learning

## 🔮 Future Enhancements

- [ ] **Transformer Architecture**: Implement Vision Transformer + GPT architecture
- [ ] **Multi-lingual Support**: Generate captions in multiple languages
- [ ] **Video Captioning**: Extend to temporal sequences
- [ ] **Interactive Web Interface**: Deploy as a web application
- [ ] **Real-time Processing**: Optimize for real-time caption generation
- [ ] **Style Transfer**: Generate captions in different styles (poetic, technical, etc.)
- [ ] **Visual Question Answering**: Extend to answer questions about images

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team**: For the excellent deep learning framework
- **MS COCO Dataset**: For providing high-quality image-caption pairs
- **ResNet Authors**: For the robust CNN architecture
- **Attention Mechanism**: Based on Bahdanau et al. (2014)
- **Research Community**: For continuous advancement in multimodal AI

---

⭐ **Star this repository if you found it helpful!**
