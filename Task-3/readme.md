# Car Color Detection Model

## Description
This project predicts the color of cars in traffic, counts cars and people at a traffic signal, and displays the results using a graphical user interface (GUI). Cars detected as blue are highlighted with red rectangles, while other colors are highlighted with blue rectangles.

## Features
- Car color detection
- Counting the number of cars and people
- GUI to upload and preview images

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd car-color-detection
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
Run the GUI:
```bash
python main.py --train --dataset stanford_cars --model car_color_model.h5
python main.py --model car_color_model.h5
python gui.py
```
Upload an image, click "Run Detection" to see the results.
