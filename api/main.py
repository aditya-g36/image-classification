from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Serve static files (frontend files)
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Setup templates for rendering HTML
templates = Jinja2Templates(directory="../frontend")

# Set up CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000/",
    "http://127.0.0.1:8000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/model_name.h5")

CLASS_NAMES = ["Bacterial_spot", "Early_Blight", "Late_Blight" ,"Leaf_Mold",
"Septoria_leaf_spot","Spider_mites_Two_spotted_spider_mite",
"Target_Spot","YellowLeaf__Curl_Virus","mosaic_virus","Healthy"]

history = np.load("../training_history.npy", allow_pickle='TRUE').item()

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    accuracy = history['accuracy']
    loss = history['loss']
    val_accuracy = history['val_accuracy']
    val_loss = history['val_loss']

    return {
        'model': int(predicted_class_index),
        'class': predicted_class,
        'confidence': float(confidence),
        'accuracy': accuracy,
        'loss': loss,
        'val_accuracy': val_accuracy,
        'val_loss': val_loss
    }

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/disease", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("diseases.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
