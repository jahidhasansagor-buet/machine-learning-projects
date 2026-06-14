import json
import torch
import io
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

CHECKPOINT = 'google/vit-base-patch16-224'

# load class names
with open('artifacts/class_names.json') as f:
    CLASS_NAMES = json.load(f)

# load the image processor from HuggingFace (needs internet first time)
image_processor = ViTImageProcessor.from_pretrained(CHECKPOINT)

# rebuild the model with the same architecture used during training
model = ViTForImageClassification.from_pretrained(
    CHECKPOINT,
    num_labels=len(CLASS_NAMES),
    ignore_mismatched_sizes=True
)

# load our trained weights — map_location='cpu' is required here
# because the weights were saved from a GPU machine (Colab/Kaggle)
model.load_state_dict(
    torch.load('artifacts/vit_brain_tumor.pt', map_location='cpu')
)
model.eval()

print(f"Model loaded. Classes: {CLASS_NAMES}")


def predict(image_bytes: bytes) -> dict:
    image  = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    inputs = image_processor(images=image, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    probs    = torch.softmax(outputs.logits, dim=-1)[0]
    pred_idx = probs.argmax().item()

    return {
        'label':      CLASS_NAMES[pred_idx],
        'confidence': round(probs[pred_idx].item(), 4)
    }
