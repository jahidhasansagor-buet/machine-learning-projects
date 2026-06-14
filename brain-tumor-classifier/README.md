# Brain Tumor MRI Classifier

## Overview
A Vision Transformer (ViT) fine-tuned to classify brain MRI scans into four
categories: glioma, meningioma, pituitary tumor, and no tumor. The model is
trained on Kaggle (T4 GPU) and served for inference through a FastAPI backend
running locally on CPU.

## Model
- Architecture: `google/vit-base-patch16-224` (fine-tuned)
- Dataset: Brain Tumor MRI Dataset (Kaggle — masoudnickparvar/brain-tumor-mri-dataset)
- Classes: glioma, meningioma, notumor, pituitary
- Test Accuracy: **95.00%**
- Epochs: 6

## Class Descriptions
| Class       | Description                                      |
|-------------|--------------------------------------------------|
| glioma      | Malignant tumor originating in glial cells       |
| meningioma  | Tumor arising from the meninges (usually benign) |
| pituitary   | Tumor of the pituitary gland                     |
| notumor     | Healthy brain scan, no tumor present             |

## API Endpoints
| Method | Endpoint  | Description                          |
|--------|-----------|--------------------------------------|
| GET    | /health   | Server status and model info         |
| POST   | /predict  | Upload MRI image, get prediction     |

## Prediction Response
```json
{
  "label": "glioma",
  "confidence": 0.9821
}
```

## Screenshots
### Swagger UI
![Swagger](screenshots/swagger.png)

## Installation
```bash
git clone <your-repo-url>
cd brain-tumor-classifier
pip install -r requirements.txt
```
Place your trained weights `vit_brain_tumor.pt` in the `artifacts/` folder
(it is not committed to git because of its size).

## Run
```bash
fastapi dev main.py
# Open http://localhost:8000/docs
```

## Technologies Used
- Python
- PyTorch, HuggingFace Transformers
- FastAPI
- Pillow
