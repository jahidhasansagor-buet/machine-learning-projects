from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import predictor

app = FastAPI(title="Brain Tumor MRI Classifier")


class PredictionOutput(BaseModel):
    label: str
    confidence: float


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model':  'ViT fine-tuned on Brain Tumor MRI',
        'classes': predictor.CLASS_NAMES
    }


@app.post('/predict', response_model=PredictionOutput)
async def predict(file: UploadFile = File(...)):
    # basic check — only accept image files
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    result      = predictor.predict(image_bytes)
    return PredictionOutput(**result)


@app.get('/', response_class=HTMLResponse)
def home():
    # simple HTML page for manual testing without Swagger
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Brain Tumor MRI Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 60px auto; padding: 0 20px; }
            h1   { color: #2c3e50; }
            input[type=file] { margin: 15px 0; display: block; }
            button { background: #2980b9; color: white; border: none; padding: 10px 24px;
                     font-size: 15px; border-radius: 5px; cursor: pointer; }
            button:hover { background: #1c6391; }
            #result { margin-top: 25px; padding: 15px; background: #ecf0f1; border-radius: 6px;
                      display: none; font-size: 16px; }
            .label      { font-size: 22px; font-weight: bold; color: #27ae60; }
            .confidence { color: #7f8c8d; margin-top: 5px; }
        </style>
    </head>
    <body>
        <h1>🧠 Brain Tumor MRI Classifier</h1>
        <p>Upload an MRI brain scan image and the model will classify it.</p>
        <p>Classes: <strong>glioma, meningioma, pituitary, notumor</strong></p>
        <input type="file" id="fileInput" accept="image/*">
        <button onclick="predict()">Predict</button>
        <div id="result">
            <div class="label" id="label"></div>
            <div class="confidence" id="confidence"></div>
        </div>
        <p style="margin-top:40px; color:#aaa;">
            For full API docs go to <a href="/docs">/docs</a>
        </p>
        <script>
            async function predict() {
                const file = document.getElementById('fileInput').files[0];
                if (!file) { alert('Please select an image file first'); return; }

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const res  = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await res.json();

                    document.getElementById('label').textContent      = 'Result: ' + data.label;
                    document.getElementById('confidence').textContent  = 'Confidence: ' + (data.confidence * 100).toFixed(2) + '%';
                    document.getElementById('result').style.display    = 'block';
                } catch (err) {
                    alert('Error: ' + err.message);
                }
            }
        </script>
    </body>
    </html>
    """
    return html
