from fastai.vision.all import (
    load_learner,
    PILImage,
)
import urllib.request
import streamlit as st
    
MODEL_URL = "https://huggingface.co/spaces/pimThrada/Dog-emotion/resolve/main/dogemotion-model-final.pkl"
urllib.request.urlretrieve(MODEL_URL, "dogemotion-model-final.pkl")
learn_inf = load_learner('dogemotion-model-final.pkl')
    
def get_image_from_upload():
    uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
    if uploaded_file is not None:
        st.image(PILImage.create((uploaded_file)))
        return PILImage.create((uploaded_file))
    return None
    
    
def predict(learn, img):
    pred, pred_idx, pred_prob = learn.predict(img)
    st.success(f"This is {pred} dog with the probability of {pred_prob[pred_idx]*100:.02f}%")
    #st.image(img, use_column_width=True)
    

def main():
    st.title('Dog emotion classification model')
    #model = load_model()
    #categories = load_labels()
    #image = load_image()
    image = get_image_from_upload()
    result = st.button('Classify')
    if result:
        predict(learn_inf, image)


if __name__ == '__main__':
    main()
