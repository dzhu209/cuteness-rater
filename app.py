import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

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

def load_model(model_path, weights_path):
    with open(model_path, 'r') as json_file:
        model_json = json_file.read()
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(weights_path)
    return model

def preprocess_image(image):
    image = image.resize((64, 64))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.reshape(image_array, (1, 64, 64, 3))
    return image_array

def make_prediction(model, image_array):
    prediction = model.predict(image_array)
    return prediction[0][0]

def main():
    st.title("Welcome to the pet photo cuteness rater!")

    # Inserting the home image
    home_image_path = "web_image/Picture_home.jpg"
    st.image(home_image_path, use_column_width=True)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        model_names = ["rater", "focus", "eye", "face", "near", "action", "accessory", "occlusion", "blur"]
        models = [
            load_model(rater_model_path, rater_weights_path),
            load_model(focus_model_path, focus_weights_path),
            load_model(eye_model_path, eye_weights_path),
            load_model(face_model_path, face_weights_path),
            load_model(near_model_path, near_weights_path),
            load_model(action_model_path, action_weights_path),
            load_model(accessory_model_path, accessory_weights_path),
            load_model(occlusion_model_path, occlusion_weights_path),
            load_model(blur_model_path, blur_weights_path)
        ]

        preprocessed_image = preprocess_image(image)
        predictions = [make_prediction(model, preprocessed_image) for model in models]

        for i, model_name in enumerate(model_names):
            prediction_value = predictions[i]
            # st.write(f"Prediction for {model_name}: {prediction_value}")

            if model_name == "rater":
                if prediction_value < 10:
                    star_image_path = "web_image/star1.png"
                elif prediction_value < 30:
                    star_image_path = "web_image/star2.png"
                elif prediction_value < 50:
                    star_image_path = "web_image/star3.png"
                elif prediction_value < 70:
                    star_image_path = "web_image/star4.png"
                else:
                    star_image_path = "web_image/star5.png"
                st.image(star_image_path, use_column_width=True)

            elif model_name == "focus":
                if prediction_value < 0.5:
                    st.write("Subject Focus: Consider to improve concentration!")
                else:
                    st.write("Subject Focus: pretty good!")

            elif model_name == "eye":
                if prediction_value < 0.5:
                    st.write("Eyes: consider to enhance eyes!")
                else:
                    st.write("Eyes: She/he has pretty eyes!")

            elif model_name == "face":
                if prediction_value < 0.5:
                    st.write("Face: Show a bit more of his/her pretty face.")
                else:
                    st.write("Face: She/he has a pretty face!")

            elif model_name == "near":
                if prediction_value < 0.5:
                    st.write("Near: Move your camera a bit closer.")
                else:
                    st.write("Near: The distance is perfect!")

            elif model_name == "action":
                if prediction_value < 0.5:
                    st.write("Action: Consider to capture his/her action.")
                else:
                    st.write("Action: She/He action is so cute!")

            elif model_name == "accessory":
                if prediction_value < 0.5:
                    st.write("Accessory: Adding some accessory can usually enhance your clicks.")
                else:
                    st.write("Accessory: Her/his is so cute!")

            elif model_name == "occlusion":
                if prediction_value < 0.5:
                    st.write("Occlusion: Great job! No blocks in front of the pet.")
                else:
                    st.write("Occlusion: Consider remove any blocks.")

            elif model_name == "blur":
                if prediction_value < 0.5:
                    st.write("Blur: Great job! The photo seems pretty clear!")
                else:
                    st.write("Blur: The photo seems a bit blur, please enhance the clearness.")

    
    # Displaying the images
    st.markdown("## Below are some images you can download and upload to try the tool:")

    image_paths = [
        "web_image/0a3faad2084666073e567516f0cc3ae5.jpg",
        "web_image/0a21bc46d78381dd90cc6ccb504ee064.jpg",
        "web_image/0a420bbb004235d2cd22ca0f068c3ae9.jpg",
        "web_image/0aa9b55eb2facee280cf1c620b5cd460.jpg"
    ]

    # Display the images in a single line
    images_style = "display: flex; justify-content: space-between;"
    st.markdown(f'<div style="{images_style}">', unsafe_allow_html=True)

    for path in image_paths:
        st.image(path, use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()





