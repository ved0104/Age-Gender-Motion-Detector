# main.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Global image size for both training and detection
IMG_SIZE = 64

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_dataset(dataset_dir):
    data = {}
    for phase in ['train', 'test']:
        images = []
        labels = []
        phase_dir = os.path.join(dataset_dir, phase)
        for root, dirs, files in os.walk(phase_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    # Heuristic: use the average colour of the image
                    mean_color = cv2.mean(img)[:3]  # (B, G, R)
                    if mean_color[0] > mean_color[1] and mean_color[0] > mean_color[2]:
                        label = 1  # blue
                    else:
                        label = 0  # other
                    images.append(img)
                    labels.append(label)
        images = np.array(images, dtype="float32") / 255.0
        labels = to_categorical(np.array(labels), num_classes=2)
        data[phase] = (images, labels)
        print(f"Loaded {len(images)} images for '{phase}'")
    return data['train'], data['test']

def train_model(dataset_dir, model_save_path='car_color_model.h5', epochs=10, batch_size=32):
    (X_train, y_train), (X_test, y_test) = load_dataset(dataset_dir)
    model = build_model()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test))
    
    # Evaluate using various metrics
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nEvaluation Metrics on Test Data:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Optionally, plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title("Loss")
    plt.legend()
    plt.show()

    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    return model

def detect_cars(image, model, min_area=500):
    detected_boxes = []
    proc = image.copy()
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            # Prepare the candidate ROI for classification
            car_roi = image[y:y+h, x:x+w]
            try:
                car_roi_resized = cv2.resize(car_roi, (IMG_SIZE, IMG_SIZE))
            except Exception as e:
                continue
            car_roi_resized = car_roi_resized.astype("float32") / 255.0
            car_roi_resized = np.expand_dims(car_roi_resized, axis=0)
            pred = model.predict(car_roi_resized)
            label = int(np.argmax(pred, axis=1)[0])
            detected_boxes.append((x, y, w, h, label))
    return detected_boxes

def detect_people(image, min_area=800):
    detected_people = []
    proc = image.copy()
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / float(w)
            if aspect_ratio > 1.5 and h > 30:  # heuristic for vertical shape
                detected_people.append((x, y, w, h))
    return detected_people
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Car Colour Detection and People Counting")
    parser.add_argument('--train', action='store_true', help='Train the car colour classification model.')
    parser.add_argument('--dataset', type=str, default='stanford_cars',
                        help='Path to the Stanford Cars dataset directory (with "train" and "test" folders).')
    parser.add_argument('--model', type=str, default='car_color_model.h5', help='File to save/load the model.')
    args = parser.parse_args()
    
    if args.train:
        train_model(args.dataset, model_save_path=args.model)
    else:
        if not os.path.exists(args.model):
            print(f"Model file '{args.model}' not found. Please train the model first (--train).")
            exit(1)
        model = load_model(args.model)
        test_img_path = 'test.jpg'  
        image = cv2.imread(test_img_path)
        if image is None:
            print("Test image not found.")
            exit(1)
        
        car_boxes = detect_cars(image, model)
        people_boxes = detect_people(image)
        
        # Draw rectangles: red for blue cars, blue for others.
        for (x, y, w, h, label) in car_boxes:
            if label == 1:
                color = (0, 0, 255)  # Red rectangle for blue cars
            else:
                color = (255, 0, 0)  # Blue rectangle for other colours
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Draw detected people in green and display count.
        for (x, y, w, h) in people_boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"People count: {len(people_boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the result.
        cv2.imshow("Detection Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
