from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import os

rater_model_path = 'rater_model.json'
rater_weights_path = 'rater_model_weights.h5'
focus_model_path = 'model_focus.json'
focus_weights_path = 'model_focus_weights.h5'
eye_model_path = 'model_eye.json'
eye_weights_path = 'model_eye_weights.h5'
face_model_path = 'model_face.json'
face_weights_path = 'model_face_weights.h5'
near_model_path = 'model_near.json'
near_weights_path = 'model_near_weights.h5'
action_model_path = 'model_action.json'
action_weights_path = 'model_action_weights.h5'
accessory_model_path = 'model_accessory.json'
accessory_weights_path = 'model_accessory_weights.h5'
occlusion_model_path = 'model_occlusion.json'
occlusion_weights_path = 'model_occlusion_weights.h5'
blur_model_path = 'model_blur.json'
blur_weights_path = 'model_blur_weights.h5'


with open(rater_model_path, 'r') as json_file:
    rater_model_json = json_file.read()
    rater_model = tf.keras.models.model_from_json(rater_model_json)
    rater_model.load_weights(rater_weights_path)

with open(focus_model_path, 'r') as json_file:
    focus_model_json = json_file.read()
    focus_model = tf.keras.models.model_from_json(focus_model_json)
    focus_model.load_weights(focus_weights_path)

with open(eye_model_path, 'r') as json_file:
    eye_model_json = json_file.read()
    eye_model = tf.keras.models.model_from_json(eye_model_json)
    eye_model.load_weights(eye_weights_path)

with open(face_model_path, 'r') as json_file:
    face_model_json = json_file.read()
    face_model = tf.keras.models.model_from_json(face_model_json)
    face_model.load_weights(face_weights_path)

with open(near_model_path, 'r') as json_file:
    near_model_json = json_file.read()
    near_model = tf.keras.models.model_from_json(near_model_json)
    near_model.load_weights(near_weights_path)

with open(action_model_path, 'r') as json_file:
    action_model_json = json_file.read()
    action_model = tf.keras.models.model_from_json(action_model_json)
    action_model.load_weights(action_weights_path)

with open(accessory_model_path, 'r') as json_file:
    accessory_model_json = json_file.read()
    accessory_model = tf.keras.models.model_from_json(accessory_model_json)
    accessory_model.load_weights(accessory_weights_path)

with open(occlusion_model_path, 'r') as json_file:
    occlusion_model_json = json_file.read()
    occlusion_model = tf.keras.models.model_from_json(occlusion_model_json)
    occlusion_model.load_weights(occlusion_weights_path)

with open(blur_model_path, 'r') as json_file:
    blur_model_json = json_file.read()
    blur_model = tf.keras.models.model_from_json(blur_model_json)
    blur_model.load_weights(blur_weights_path)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']
        
        # Read the image file
        img = Image.open(file.stream)
        
        # Preprocess the image
        img = img.resize((64, 64))
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize pixel values between 0 and 1
        
        # Reshape the image array to match the input shape of your model
        img_array = np.reshape(img_array, (1, 64, 64, 3))
        
        # Make the prediction
        prediction_rater = rater_model.predict(img_array)
        prediction_focus = focus_model.predict(img_array)
        prediction_eye = eye_model.predict(img_array)
        prediction_face = face_model.predict(img_array)
        prediction_near = near_model.predict(img_array)
        prediction_action = action_model.predict(img_array)
        prediction_accessory = accessory_model.predict(img_array)
        prediction_occlusion = occlusion_model.predict(img_array)
        prediction_blur = blur_model.predict(img_array)

        # Return the prediction result
        return render_template('result.html', prediction_rater=prediction_rater, prediction_focus=prediction_focus, prediction_eye=prediction_eye,prediction_face=prediction_face, prediction_near=prediction_near, prediction_action=prediction_action, prediction_accessory=prediction_accessory, prediction_occlusion=prediction_occlusion,prediction_blur=prediction_blur)
        
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def show_result():
    # Retrieve the necessary data or results to display on the result.html page
    # ...
    # Replace this with your code to retrieve the necessary data or results

    # Render the result.html page with the required data or results
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)


