from fastai.vision.all import (
    load_learner,
    PILImage,
)
import urllib.request
import streamlit as st
    
MODEL_URL = "https://huggingface.co/spaces/pimThrada/Dog-emotion/resolve/main/dogemotion-model-5-final.pkl"
urllib.request.urlretrieve(MODEL_URL, "dogemotion-model-5-final.pkl")
learn_inf = load_learner('dogemotion-model-5-final.pkl')

st.title('Dog emotion classification model')

tab1, tab2 = st.tabs(["uploaded file", "Take a picture"])

with tab1:
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            st.image(PILImage.create((uploaded_file)))
            return PILImage.create((uploaded_file))
        return None
        
    def predict(learn, img):
        pred, pred_idx, pred_prob = learn.predict(img)
        st.success(f"{pred} {pred_prob[pred_idx]*100:.02f}%")

    image = get_image_from_upload()
    result1 = st.button('Classify',key=0)
    if result1:
        predict(learn_inf, image)
    

with tab2:
    def take_a_picture():
        picture = st.camera_input("Take a picture")
        if picture:
            st.image(picture)
            return PILImage.create((picture)) 
        return None

    def predict(learn, img):
        pred, pred_idx, pred_prob = learn.predict(img)
        if pred=='unknow':
            st.success(f"This is {pred} with the probability of {pred_prob[pred_idx]*100:.02f}%")
        else:
            st.success(f"This is {pred} dog with the probability of {pred_prob[pred_idx]*100:.02f}%")

    image = take_a_picture()
    result = st.button('Classify',key=1)
    if result:
        predict(learn_inf, image)





