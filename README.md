# Visual Question Answering (VQA) Pipeline

A Python project for Visual Question Answering (VQA) using Vision-Language Models (VLMs) powered by PyTorch and Hugging Face Transformers.

## 📁 Project Structure

```text
vqa-vlm-pipeline/
├── notebooks/
│   └── vqa_pipeline.ipynb   # End-to-end notebook (EDA, batch inference, EM accuracy)
├── src/
│   ├── data_loader.py       # Custom Dataset, DataLoader, HF dataset loading, and EDA
│   └── model.py             # Vision-Language Model wrapper (VQAClassifier with BLIP)
├── .gitignore               # Ignored cache, environment, and IDE files
├── README.md                # Project documentation
└── requirements.txt         # Project dependencies
```

## 🛠️ Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/suyashpjadhav/vqa-vlm-pipeline.git
   cd vqa-vlm-pipeline
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows (PowerShell):
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Using Python Modules

```python
from src.data_loader import load_vqa_subset, run_eda
from src.model import VQAClassifier

# 1. Load dataset & run EDA
vqa_data = load_vqa_subset(split="validation[:50]")
run_eda(vqa_data, num_samples=3)

# 2. Initialize VQA Classifier (auto-detects CUDA / CPU)
model = VQAClassifier(model_name="Salesforce/blip-vqa-base")

# 3. Predict answer for a single image & question
sample = vqa_data[0]
answer = model.answer_question(sample['image'], sample['question'])
print("Predicted Answer:", answer)
```

### Jupyter Notebook
Explore [`notebooks/vqa_pipeline.ipynb`](file:///c:/Users/BIT/Desktop/vqa-vlm-pipeline/notebooks/vqa_pipeline.ipynb) for the interactive pipeline including EDA, batch inference, and exact match (EM) accuracy evaluation.

---

## 🌐 Replicating on Kaggle

Follow these steps to run this pipeline on a **Kaggle Notebook** with GPU acceleration:

### Step 1: Create a Kaggle Notebook
1. Log in to [Kaggle](https://www.kaggle.com/) and click **+ Create -> New Notebook**.
2. In the right panel under **Notebook Settings**:
   - Set **Accelerator** to **GPU P100** or **GPU T4 x2**.
   - Ensure **Internet** is turned **On** (required to download Hugging Face models and datasets).

### Step 2: Clone the Repository & Install Dependencies
In the first code cell of your Kaggle Notebook, run:

```bash
# Clone the project repository
!git clone https://github.com/suyashpjadhav/vqa-vlm-pipeline.git
%cd vqa-vlm-pipeline

# Install required dependencies
!pip install -r requirements.txt
```

### Step 3: Run the VQA Pipeline
In the subsequent cell, import the modules and execute the end-to-end workflow:

```python
import sys
import os

# Add src to Python path
sys.path.append(os.path.abspath('src'))

from data_loader import load_vqa_subset, run_eda
from model import VQAClassifier

# 1. Load dataset & display EDA plots
vqa_data = load_vqa_subset(split='validation[:50]')
run_eda(vqa_data, num_samples=3)

# 2. Initialize VQAClassifier (automatically leverages Kaggle GPU)
model = VQAClassifier(model_name="Salesforce/blip-vqa-base")

# 3. Run batch prediction
images = [s['image'] for s in vqa_data]
questions = [s['question'] for s in vqa_data]
predictions = model.batch_predict(images[:8], questions[:8])

print("Sample Predictions:", predictions)
```

### Step 4: Import Existing Notebook (Alternative)
Alternatively, you can upload [`notebooks/vqa_pipeline.ipynb`](file:///c:/Users/BIT/Desktop/vqa-vlm-pipeline/notebooks/vqa_pipeline.ipynb) directly to Kaggle:
1. In Kaggle Notebook, go to **File -> Import Notebook**.
2. Upload `vqa_pipeline.ipynb`.
3. Set Accelerator to **GPU T4** and run all cells sequentially.

---

## 📜 License
MIT
