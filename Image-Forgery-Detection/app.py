# import keras
# from keras import backend as K
#import model
import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, redirect, request, render_template
import matplotlib.pyplot as plt
import base64
import os
import io

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

app = Flask(__name__)

# def decode_an_image_array(rgb, dn=1):
#     x = np.expand_dims(rgb.astype('float32') / 255. * 2 - 1, axis=0)[:, ::dn, ::dn]
#     K.clear_session()
#     manTraNet = model.load_trained_model()
#     return manTraNet.predict(x)[0, ..., 0]


# def decode_an_image_file(image_file, dn=1):
#     mask = decode_an_image_array(image_file, dn)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(image_file[::dn, ::dn])
#     plt.imshow(mask, cmap='jet', alpha=.5)
#     plt.savefig('h.png', bbox_inches='tight', pad_inches=-0.1)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Image Prediction Steps

class_names = ['fake', 'real']

## Import Libraries
from PIL import Image, ImageChops, ImageEnhance
import os
import itertools
import base64
from keras.models import load_model
# Load the model
path='Resnet100epoch.h5'
model_path = "C:/Users/hp/OneDrive/Desktop/seaoff/Image-Forgery-Detection/models/"+path
model = load_model(model_path,compile=False)
# Function to convert into ELA
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.jpg'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image
# Prepare Image
image_size = (128, 128)
def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() /255.0
#Predict Step
def predict(real_image_path):
    image = prepare_image(real_image_path)
    image = image.reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    y_pred_class = np.argmax(y_pred, axis = 1)[0]
    x= class_names[y_pred_class]
    y = round(np.amax(y_pred) * 100, 3)
    print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')
    return x,y

@app.route("/", methods=['GET', 'POST'])
def base():
    if request.method == 'GET':
        return render_template("base.html", output=0)
    else:
        if 'input_image' not in request.files:
            print("No file part")
            return redirect(request.url)

        file = request.files['input_image']

        if file.filename == '':
            print('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # path="C:/Users/hp/OneDrive/Desktop/seaoff/Image-Forgery-Detection/static/uploads"
            # real_image_path = '../static/uploads/'  # Set the path where you want to save the uploaded image
            # #file.save(file)
            img_bytes = io.BytesIO()
            file.save(img_bytes)

            # Convert the image bytes to base64
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

            # Get the file extension
            _, file_extension = os.path.splitext(file.filename)
            file_extension = file_extension[1:].lower()  # Remove the dot and convert to lowercase
            class1, confidence = predict(file)
            # Pass the base64-encoded image and file extension to the template
            return render_template("base.html", class_name=class1, confidence=confidence, img_base64=img_base64, file_extension=file_extension, output=1)

            
            # print(class1,confidence)
            # return render_template("base.html", class_name=class1, confidence=confidence,img=file, output=1)



if __name__ == "__main__":
    #app.secret_key = 'qwertyuiop1234567890'
    #port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0')
    
