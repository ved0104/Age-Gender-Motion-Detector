# gui.py
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

class GenderDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gender Detection with Hair Logic")        
        # Load model with custom objects
        self.model = tf.keras.models.load_model(
            "C:/Users/dubey/OneDrive/Desktop/Coding/Projects/AgeGenderDetector/Task-2/gender_age_hair_model.h5",
            custom_objects={
                'mse': tf.keras.losses.MeanSquaredError(),
                'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
                'mae': tf.keras.metrics.MeanAbsoluteError(),
                'accuracy': tf.keras.metrics.BinaryAccuracy()
            }
        )
        self.create_widgets()
    
    def create_widgets(self):
        # Upload Button
        self.upload_btn = tk.Button(
            self.root, 
            text="Upload Image", 
            command=self.upload_image
        )
        self.upload_btn.pack(pady=20)
        
        # Image Display
        self.image_panel = tk.Label(self.root)
        self.image_panel.pack()
        
        # Result Label
        self.result_label = tk.Label(
            self.root, 
            text="", 
            font=("Arial", 14)
        )
        self.result_label.pack(pady=10)
    
    def preprocess_image(self, image_path):
        img = Image.open(image_path).resize((128, 128))
        img = np.array(img) / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    
    def predict_gender(self, image_path):
        img = self.preprocess_image(image_path)
        age_pred, gender_pred, hair_pred = self.model.predict(img)
        
        if 20 <= age_pred <= 30:
            gender = "Female" if hair_pred > 0.5 else "Male"
        else:
            gender = "Female" if gender_pred > 0.5 else "Male"
        
        return f"Predicted Gender: {gender}, Age: {int(age_pred)}"
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            # Display image
            img = Image.open(file_path)
            img = img.resize((200, 200))
            img = ImageTk.PhotoImage(img)
            self.image_panel.config(image=img)
            self.image_panel.image = img
            
            # Show prediction
            result = self.predict_gender(file_path)
            self.result_label.config(text=result)

if __name__ == "__main__":
    root = tk.Tk()
    app = GenderDetectionApp(root)
    root.mainloop()