# NBME_Daberta_v1 【train】

A deep learning solution for automatically scoring clinical patient notes using DeBERTa transformer model. This project implements a token-level binary classification approach to identify relevant clinical features in patient notes.

## 🎯 Project Overview

This project tackles the NBME (National Board of Medical Examiners) challenge of scoring clinical patient notes. The goal is to automatically identify and extract specific clinical features mentioned in patient notes, which is crucial for medical education and assessment.

### Key Features

- **Token-level Classification**: Uses DeBERTa-base model for fine-grained token classification
- **5-Fold Cross Validation**: Robust evaluation with GroupKFold to prevent data leakage
- **Automatic Annotation Fixing**: Intelligent correction of misaligned annotations
- **Threshold Optimization**: Dynamic threshold tuning for optimal F1 score
- **Memory Optimization**: Efficient memory management for large-scale training

## 📊 Model Architecture

- **Base Model**: microsoft/deberta-base
- **Task**: Token-level binary classification
- **Sequence Length**: 512 tokens
- **Input**: [Feature Text] + [Patient Note]
- **Output**: Binary probability for each token

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch transformers sklearn pandas numpy tqdm kagglehub
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nbme-deberta-scoring.git
cd nbme-deberta-scoring
```

2. Download the dataset:
```python
import kagglehub
nbme_score_clinical_patient_notes_path = kagglehub.competition_download('nbme-score-clinical-patient-notes')
```

### Usage

1. **Basic Training**:
```bash
python nbme_deberta_v1_train.py
```

2. **Debug Mode** (for development):
```python
# In the script, set:
CFG.debug_mode = True
CFG.run_single_fold = True  # For quick testing
```

## ⚙️ Configuration

Key hyperparameters in the `CFG` class:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_name` | "microsoft/deberta-base" | Pre-trained model |
| `max_len` | 512 | Maximum sequence length |
| `batch_size` | 8 | Training batch size |
| `lr` | 1e-5 | Learning rate |
| `epochs` | 4 | Training epochs |
| `n_folds` | 5 | Cross-validation folds |

## 📈 Performance

The model achieves competitive performance with:
- **Evaluation Metric**: Micro F1-score at character level
- **Cross Validation**: 5-fold GroupKFold
- **Threshold Optimization**: Dynamic threshold tuning per fold

## 🔧 Key Components

### Data Processing
- **Annotation Fixing**: Automatic correction of misaligned spans
- **Character-to-Token Mapping**: Precise alignment between character and token positions
- **Quality Checks**: Comprehensive data validation

### Model Training
- **Mixed Precision**: Optional FP16 training for memory efficiency
- **Gradient Checkpointing**: Memory optimization for large models
- **Cosine Scheduling**: Learning rate scheduling with warmup

### Evaluation
- **Span Reconstruction**: Converting token probabilities back to character spans
- **Micro F1**: Character-level micro F1 evaluation
- **Threshold Search**: Grid search for optimal classification threshold

## 📁 Project Structure

```
nbme-deberta-scoring/
├── nbme_deberta_v1_train.py    # Main training script
├── nbme_ckpt/                  # Model checkpoints
│   ├── fold0.pt
│   ├── fold1.pt
│   └── ...
├── cfg.json                    # Configuration backup
└── README.md
```

## 🔍 Model Details

### Architecture
```python
class DebertaForTokenBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/deberta-base")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 1)
```

### Loss Function
- **BCEWithLogitsLoss**: Binary cross-entropy for token classification
- **Class Imbalance Handling**: Automatic threshold optimization

## 📊 Evaluation Metrics

The primary evaluation metric is **micro F1-score** calculated at the character level:

```python
def compute_micro_f1(pred_df):
    tp = fp = fn = 0
    for ground_truth, prediction in zip(pred_df.ground, pred_df.pred):
        # Convert spans to character sets and compute intersection
        # F1 = 2*TP / (2*TP + FP + FN)
```

## 🛠️ Advanced Features

### Memory Optimization
- Gradient checkpointing for reduced memory usage
- Mixed precision training (optional)
- Efficient data loading with proper cleanup

### Robust Training
- Cross-validation with group splitting (by patient number)
- Automatic annotation correction
- Dynamic threshold optimization per fold

### Debug Mode
- Comprehensive data quality checks
- Label creation verification
- Training progress monitoring

## 📝 Usage Examples

### Custom Configuration
```python
class CFG:
    model_name = "microsoft/deberta-base"
    max_len = 512
    batch_size = 16  # Adjust based on GPU memory
    lr = 2e-5        # Higher learning rate
    epochs = 5       # More epochs
```

### Inference
```python
# Load trained model
model = DebertaForTokenBinary()
checkpoint = torch.load('nbme_ckpt/fold0.pt')
model.load_state_dict(checkpoint['model_state_dict'])
threshold = checkpoint['best_threshold']

# Make predictions
with torch.no_grad():
    logits = model(**batch)["logits"].sigmoid()
    # Apply threshold and reconstruct spans
```


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NBME for providing the clinical notes dataset
- Hugging Face for the transformers library
- Microsoft for the DeBERTa model
- Kaggle community for insights and discussions
