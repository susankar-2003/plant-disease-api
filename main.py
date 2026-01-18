from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import os
import uuid
import json
from typing import List  # <--- NEW IMPORT

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
MODEL_FILE = "best_model.tflite"
LABELS_FILE = "labels.txt"
INFO_FILE = "disease_info.json"
UPLOAD_DIR = "collected_images"
CONFIDENCE_THRESHOLD = 98.0

# --- LOAD RESOURCES ---
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(LABELS_FILE, 'r') as f:
        class_names = f.read().splitlines()

    with open(INFO_FILE, 'r') as f:
        disease_info = json.load(f)
        
    print(f"✅ Model & Data Loaded Successfully")

except Exception as e:
    print(f"❌ Error loading files: {e}")

# --- HELPER FUNCTIONS ---
def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_shape = input_details[0]['shape']
    image = image.resize((input_shape[2], input_shape[1]))
    img_array = np.array(image)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch.astype(np.float32)

def save_image_locally(image_bytes, folder_name):
    try:
        directory = os.path.join(UPLOAD_DIR, folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = f"{uuid.uuid4()}.jpg"
        file_path = os.path.join(directory, filename)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
    except Exception as e:
        print(f"Failed to save: {e}")

@app.get("/")
def index():
    return {"message": "Multi-Plant API is Online!"}

# --- UPDATED PREDICTION ENDPOINT ---
@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)): # Accepts a LIST of files
    results_list = []

    for file in files:
        try:
            # 1. Read and Process
            image_bytes = await file.read()
            img_batch = process_image(image_bytes)

            # 2. Predict
            interpreter.set_tensor(input_details[0]['index'], img_batch)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            raw_output = output_data[0]
            if np.abs(1.0 - np.sum(raw_output)) < 0.1:
                scores = raw_output
            else:
                scores = tf.nn.softmax(raw_output).numpy()

            predicted_index = np.argmax(scores)
            confidence = float(np.max(scores)) * 100

            # 3. Logic
            if confidence < CONFIDENCE_THRESHOLD:
                result_class = "Unknown"
                display_name = f"Could not recognize ({file.filename})"
                description = "Image is unclear or unknown."
                treatment = "N/A"
                prevention = "N/A"
                save_folder = "undefined"
            else:
                result_class = class_names[predicted_index]
                save_folder = result_class
                
                info = disease_info.get(result_class, {})
                display_name = info.get("name", result_class)
                description = info.get("description", "No description.")
                treatment = info.get("treatment", "No info.")
                prevention = info.get("prevention", "No info.")

            # 4. Save
            save_image_locally(image_bytes, save_folder)

            # 5. Add to list
            results_list.append({
                "filename": file.filename,
                "disease": display_name,
                "confidence": f"{confidence:.2f}%",
                "description": description,
                "treatment": treatment,
                "prevention": prevention
            })

        except Exception as e:
            results_list.append({
                "filename": file.filename,
                "error": str(e)
            })

    return results_list # Return the list of all results

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
