from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
import os
from django.conf import settings
import json

# Load the model
# model = load_model('C:/proj/PlantDiseaseDetection/DiseaseDetectionApp/tomato_disease_model.keras')
model = load_model('C:/my/PlantDiseaseDetectionRepo/DiseaseDetectionApp/tomato_disease_model.keras')

def detect_disease(request):
    result = 'No prediction made.'
    confidence = "0"

    with open('C:/my/PlantDiseaseDetectionRepo/DiseaseDetectionApp/prevention_methods.json', 'r') as file:
        prevention_methods = json.load(file)

    context = {'result': result, 'confidence': confidence}  # Initialize context

    if request.method == "POST" and 'image_input' in request.FILES:
        # Get the uploaded file
        uploaded_file = request.FILES['image_input']

        # Define the path to save the uploaded image
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')
        os.makedirs(upload_dir, exist_ok=True)  # Create the directory if it doesn't exist
        
        # Save the file
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        try:
            # Preprocess the input image
            img = load_img(file_path, target_size=(256, 256))
            img_array = img_to_array(img) / 255.0  # Normalize the image
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make predictions
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=-1)

            # Map predictions to class labels
            class_labels = [
                'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 
                'Tomato___Late_blight', 
                'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 
                'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                'Tomato___Tomato_mosaic_virus', 
                'Tomato___healthy'
            ]
            result = f"Predicted label: {class_labels[predicted_class[0]]}"
            confidence = f"Confidence score: {predictions[0][predicted_class[0]]*100:.1f} / 100"
            disease = class_labels[predicted_class[0]]
            prevention = prevention_methods.get(disease, [])

            context = {
                'result': result,
                'confidence': confidence,
                'prevention': prevention

            }
        except Exception as e:
            result = f"Error processing the image: {e}"
            context = {
                'result': result
            }
    return render(request, 'index.html', context)

# Create your views here.
def index(request):
    return render(request,'index.html')

def register(request):
    return

def login(request):
    return