import streamlit as st
from keras.models import load_model
from PIL import Image

from utils import classify

#set title
col1, col2 = st.columns([1, 5])

# Display logo in the first column
with col1:
    st.image("model/bina.png", width=85)  # Adjust width as needed

# Display title in the second column
with col2:
    st.title('Breast Cancer Predictor')

st.header("upload an image of an ultrasound scan of a breast tissue")
#set header
st.text('By Mohammed Saabiq Saha')

#upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

#load classifier
model = load_model('model/keras_model.h5')


#load classnames
with open('model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

#display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image)

    #classify image
    class_name, conf_score = classify(image, model, class_names)

    #write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}".format(conf_score))
