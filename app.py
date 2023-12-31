#import time
import numpy as np
import streamlit as st
#import urllib.request
import tomli
#from PIL import Image
#from utils import *
#from keras.utils import load_img, img_to_array

#from keras.models import Sequential
#from keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Dropout, Dropout
#from keras.losses import sparse_categorical_crossentropy, binary_crossentropy
#from keras.optimizers import Adam
#from PIL import Image
#from keras.utils import load_img, img_to_array
import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt

#with open("config.toml", "rb") as f:
    #toml_dict = tomli.load(f)

#pc = st.get_option('theme.primaryColor')
#bc = st.get_option('theme.backgroundColor')
#sbc = st.get_option('theme.secondaryBackgroundColor')
#tc = st.get_option('theme.textColor')
def gen_labels():
    labels = {0 : "Psilocybe Azurescens", 1 : "Psilocybe Semilanceata", 2 : "Panaeolus Cinctulus"}
    return labels

def preprocess(image):
    image = np.array(image.resize((300, 300), Image.ANTIALIAS))
    image = np.array(image, dtype='uint8')
    image = np.array(image)/255.0
    return image


def CNN_Model():
    input_shape = (300, 300, 3)
    num_classes=3
    model = Sequential()    
    model.add(Conv2D(32, kernel_size=(3,3), padding='Same', input_shape = input_shape, activation='relu'))
    model.add(MaxPool2D(pool_size=2)) 

    model.add(Conv2D(32, kernel_size = (3,3), padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=2)) 

    model.add(Conv2D(32, kernel_size = (3,3), padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=2)) 
    model.add(Flatten())

    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2000,activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation = 'softmax'))
    return model  
    
labels = gen_labels()

html_temp = '''
    <div style="padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Klasifikasi Jenis Jamur</h1></center>
    </div>
    '''

hide_streamlit_style = '''
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    '''
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(html_temp, unsafe_allow_html=True)
vert_space = '<div style="padding: 180px 5px;"></div>'
html_temp = '''
    <div>
    <h2></h2>
    <center><h3>Silakan unggah citra untuk menemukan jenisnya</h3></center>
    </div>
    '''
st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(html_temp, unsafe_allow_html=True)
opt = st.selectbox("Pilih metode untuk klasifikasi citra\n", ('Silakan Pilih', 'Unggah citra via link', 'Unggah citra dari piranti'))
image = None  # Inisialisasi image dengan None

if opt == 'Unggah citra dari piranti':
    file = st.file_uploader('Pilih', type=['jpg', 'png', 'jpeg'])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is not None:
        image = Image.open(file).convert("RGB")

elif opt == 'Unggah citra via link':
    try:
        img = st.text_input('Masukkan URL Gambar...')
        image = Image.open(urllib.request.urlopen(img)).convert("RGB")
    except:
        if st.button('Submit'):
            show = st.error("Masukkan URL citra yang Benar")
            time.sleep(4)
            show.empty()

try:
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        if image is not None:
            st.image(image, use_column_width=False, width=200, caption='Uploaded Image')
            if st.button("ㅤㅤㅤㅤPredictㅤㅤㅤㅤ"):
                img = preprocess(image)
                model = CNN_Model()
                model.load_weights("model1.h5")
                prediction = model.predict(img[np.newaxis, ...])
                proba = np.max(prediction[0], axis=-1)
                prediction_idx = np.argmax(prediction[0], axis=-1)
                if prediction_idx < len(labels):
                    st.info(f'ㅤㅤ {labels[prediction_idx]}')
                    st.info(f'ㅤㅤ Akurasi = {proba:.2%}')
                else:
                    st.info('Tidak dapat mengklasifikasikan citra.')
    with col3:
        pass
except Exception as e:
    st.info(e)
    pass

