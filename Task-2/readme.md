# Gender, Age, and Hair Length Detection Project

This project uses a multi-task deep learning model to predict a person's age, gender, and hair length from an image. The model includes custom logic such that for individuals between the ages of 20 and 30, hair length is used to infer gender (i.e., long hair is treated as female and short hair as male). For persons outside this age range, the gender prediction is used directly.

## Project Structure

- **main.ipynb**: Jupyter notebook that contains the code for data loading, preprocessing, model training, evaluation, and saving.
- **gui.py**: A Tkinter-based graphical user interface (GUI) that allows users to upload an image and view predictions.
- **gender_age_hair_model.h5**: The saved Keras model (generated after training).
- **requirements.txt**: List of required Python packages.
- **README.md**: This file.

## Installation

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
