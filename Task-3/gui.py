import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import confusion_matrix

labels = [0, 1]
cm = confusion_matrix(y_true, y_pred, labels=labels)
model = load_model('car_color_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Function to make predictions
INPUT_SIZE = (64, 64)
def predict_color(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, INPUT_SIZE)  # Resize to match the model's input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)  # Assuming softmax output
    return predicted_label

# GUI Application
class CarColorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Color Detection")
        self.root.geometry("400x400")

        self.label = tk.Label(root, text="Car Color Detection Model", font=("Arial", 16))
        self.label.pack(pady=10)

        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=5)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=5)

        self.show_metrics_btn = tk.Button(root, text="Show Performance Metrics", command=self.show_metrics)
        self.show_metrics_btn.pack(pady=5)

        self.true_labels = []
        self.predicted_labels = []

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img = img.resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)

            panel = tk.Label(self.root, image=img_tk)
            panel.image = img_tk
            panel.pack()

            predicted_label = predict_color(file_path)
            self.predicted_labels.append(predicted_label)

            true_label = int(input("Enter the actual label for the image (for metrics calculation): "))
            self.true_labels.append(true_label)

            self.result_label.config(text=f"Predicted Car Color: {predicted_label}")

    def show_metrics(self):
        if self.true_labels and self.predicted_labels:
            accuracy = accuracy_score(self.true_labels, self.predicted_labels)
            precision = precision_score(self.true_labels, self.predicted_labels, average='weighted')
            recall = recall_score(self.true_labels, self.predicted_labels, average='weighted')
            f1 = f1_score(self.true_labels, self.predicted_labels, average='weighted')
            conf_matrix = confusion_matrix(self.true_labels, self.predicted_labels)

            metrics_msg = (
                f"Accuracy: {accuracy:.4f}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"F1 Score: {f1:.4f}\n"
                f"Confusion Matrix:\n{conf_matrix}"
            )
            messagebox.showinfo("Performance Metrics", metrics_msg)
        else:
            messagebox.showwarning("No Data", "Upload images and provide actual labels to calculate metrics.")

# Run the GUI
root = tk.Tk()
app = CarColorDetectionApp(root)
root.mainloop()
