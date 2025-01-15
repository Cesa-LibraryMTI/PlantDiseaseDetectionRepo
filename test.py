from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the model
model = load_model('tomato_disease_model.keras')

# Preprocess the input image
image_path = 'nigga.JPG'
img = load_img(image_path, target_size=(256, 256))  # Replace with your input size
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

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
print("Predicted label:", class_labels[predicted_class[0]])
