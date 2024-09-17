from os.path import dirname, abspath, join
import sys
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
sys.path.append(join(parent_dir, 'src'))
import loading_dataset as ld
sys.path.append(current_dir)
import os
import mediapipe as mp
from flask import Flask, render_template, request
import numpy as np
from torch import tensor, float32, argmax
from datetime import datetime
from glob import glob
from PIL import Image
import predict

DEVICE = ld.load_device()

# 
def croped_shaped_image(folder_path):
    """ Get most recent image and shaped in right size and format

    Args: folder_path : path of where the array images are saved

    Returns: Resized to (512,512) of the most recent image
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return None
    png_files = glob(os.path.join(folder_path, "*.png"))\

    if not png_files:
        print(f"No PNG files found in folder '{folder_path}'.")
        return None
    
    png_files.sort(key=os.path.getmtime, reverse=True)
    img = Image.open(png_files[0])
    
    width, height = img.size
    crop_size = min(width, height)
    crop_left = (width - crop_size) // 2 
    crop_top = (height - crop_size) // 2
    crop_right = crop_left + crop_size
    crop_bottom = crop_top + crop_size
    # Crop the image
    cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    # Resize the cropped image to 512x512
    resized_img = cropped_img.resize((512, 512))  # Use antialiasing for better qualityImage.ANTIALIAS
    return resized_img

def getLetter():
    """Give the model the most recent image to predict the letter
    
    Returns:
        Returns the letter with the highest propability
    """
    X = croped_shaped_image("images/saved_arrays")
    image_array = np.array(X.convert('RGB'))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    X_array = ld.image_to_landmarks(mp_image)
    X_array =X_array.astype(np.float32)
    mlp = predict.load_mlp_model()
    prediction = mlp(tensor(X_array.reshape(1,21,2),dtype=float32).to(DEVICE)) # reshape to incorporate batch size
    letter_prediction = argmax(prediction) # letter is of highest probability
    return predict.ALPHABET[letter_prediction]

def save_image(image_array, folder_path, filename):
    """Saves numpy array to given path as the give filenmae
    Args:
        image_array (numpy array):  image we are saving
        folder_path (path): where we are saving the image
        filename (String): filename of saved image
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image_array.astype(np.uint8))
    # Save the image as a PNG file
    image.save(os.path.join(folder_path, filename))
    print(f"Image saved successfully: {os.path.join(folder_path, filename)}")
  
app = Flask(__name__)

@app.route("/") # rendering main page
def index():
    return render_template("index.html", name='Main')

@app.route('/how-to-use') #rendering intruction page
def how_to_use():
    return render_template('how_to_use.html')

@app.route("/receive",methods=["POST"])
def receive_data():
    """ Gets a 1D array from the webcam, after resizing, and saving it we return the predicted letter
    Returns:
        String: predicted letter
    """
    data = request.get_json()
    # image array from main.jss
    image_data = data.get('imageData', [])
    image_data = np.asarray(image_data)
    image_data = image_data[:, :, :3]# (480, 640, 3)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory_path = 'images/saved_arrays'
    os.makedirs(directory_path, exist_ok=True)
    filename = f"image_data_{timestamp}.png"
    save_image(image_data, directory_path, filename)
    letter = getLetter()
    
    return {'letter': letter}

@app.route("/feedback",methods=["POST"])
def feedback():
    rating = request.form['rate']
    feedback_file = 'feedback.txt'
    with open(feedback_file, 'a') as file:
        file.write(f"{rating}\n")
    return render_template('thank_you.html', message="Thank you for your feedback!")


if __name__ == "__main__":
    app.run(port=5000,debug=True)