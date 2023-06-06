from fastai.vision.all import (
    load_learner,
    PILImage,
)
import urllib.request
import glob
import streamlit as st
    
MODEL_URL = "https://huggingface.co/spaces/pimThrada/Dog-emotion/resolve/main/dogemotion-model-5-final.pkl"
urllib.request.urlretrieve(MODEL_URL, "dogemotion-model-5-final.pkl")
learn_inf = load_learner('dogemotion-model-5-final.pkl')

def get_image_from_test_set():
    imgpath = glob.glob('data/images/*')
    imgsel = st.slider('Select random images from test set.', min_value=1, max_value=len(imgpath), step=1) 
    image_file = imgpath[imgsel-1]
    if image_file is not None:
        st.image(PILImage.create((image_file)))
        return PILImage.create((image_file))
    return None

def get_image_from_upload():
    uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
    if uploaded_file is not None:
        st.image(PILImage.create((uploaded_file)))
        return PILImage.create((uploaded_file))
    return None

def take_a_picture():
    picture = st.camera_input("Take a picture")
    if picture:
        st.image(picture)
        return PILImage.create((picture)) 
    return None
        
def predict(learn, img):
    pred, pred_idx, pred_prob = learn.predict(img)
    if pred=='unknown':
        st.success(f"This is {pred} with the probability of {pred_prob[pred_idx]*100:.02f}%")
    else:
        st.success(f"This is {pred} dog with the probability of {pred_prob[pred_idx]*100:.02f}%")
            
def main():
    st.title('Dog emotion classification model')
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ["Use image from test set", "Uploaded file", "Take a picture"])
    if datasrc == "Use image from test set":
        image = get_image_from_test_set()
    elif datasrc == "Uploaded file": 
        image = get_image_from_upload()
    else:
        image = take_a_picture()
    result1 = st.button('Classify',key=1)
    if result1:
        predict(learn_inf, image)        
    
if __name__ == '__main__':
    main()


        
