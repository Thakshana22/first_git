



'''import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st


st.header("Image Classification Model")
model = load_model('E:/ML_project/Image_Classification/Image_classify.keras') 
model=['banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

img_height =180
img_width =180

image = st.text_input('Enter Image name','corn.jpg')
#image = st.text_input('Enter Image name','Apple.jpg')

image_load = tf.keras.utils.load_img(image,target_size=(img_height,img_width))
img_arr =tf.keras.utils.array_to_img(image_load)
img_bat =tf.expand_dims(img_arr,0)


predict = model.predict(img_bat)
score = tf.nn.softmax(predict)

st.image(image,width=200)
st.write("HI!!")
st.write("I identify the fruit/veggie is  " + data_cat[np.argmax(score)])
st.write("With accuracy of "+str(np.max(score)*100))
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

# Streamlit header
st.header("Image Classification Model")

# Load the pre-trained model
model = load_model('E:/ML_project/Image_Classification/Image_classify.keras')

# Class names
class_names = [
    'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 
    'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 
    'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 
    'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Image dimensions
img_height = 180
img_width = 180

# User input for image path
image_path = st.text_input('Enter Image name', 'corn.jpg')

# Load and preprocess the image
try:
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    # Make prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])

    # Display the image
    st.image(image_path, width=200)

    # Display the prediction
    st.write("HI!!")
    st.write("I identify the fruit/veggie as: " + class_names[np.argmax(score)])
    st.write("With accuracy of: " + str(np.max(score) * 100) + "%")

except Exception as e:
    st.error(f"Error: {e}. Please check the image path and try again.")