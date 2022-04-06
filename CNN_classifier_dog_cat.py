########################################################################################################################
# TABLE OF CONTENTS

"""
import the libraries

* PART 1 - DATA PREPROCESSING
    + PREPROCESSING THE TRAINING SET
    + PREPROCESSING THE TEST SET

* PART 2 - BUILDING THE CNN
    + INITIALISING THE CNN
    + STEP 1: CONVOLUTION
    + STEP 2: POOLING
    ADDING A SECOND CONVOLUTIONAL LAYER
    + STEP 3: FLATTENING
    + STEP 4: FULL CONNECTION
    + STEP 5: OUTPUT LAYER

* PART 3 - TRAINING THE CNN
    + COMPILING THE CNN
    + TRAING THE CNN ON THE TRAINING SET AND EVALUATING IT ON THE TEST SET

* PART 4: MAKING A SINGLE PREDICTION

"""
########################################################################################################################

# CONVOLUTIONAL NEURAL NETWORK

# import the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# using tensorflow backend
tf.__version__

# PART 1 - DATA PREPROCESSING
# + preprocessing the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,               # Chuyển các pixel thành 0 và 1 (normalization - chuẩn hóa)
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),                         # giảm chiều kích thước ảnh
    batch_size=32,
    class_mode='binary'                           # chọn chế độ phân lớp binary
)

# + preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)


# PART 2 - BUILDING THE CNN
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# + step 1: convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# + step 2: pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# + Adding a second convolutional layers
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# + step 3: Flattening
cnn.add(tf.keras.layers.Flatten())

# + step 4: Full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# + step 5: output layer
# phân loại nhị phân là sigmoid, còn softmax là đa chiều
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



# PART 3 - TRAINING THE CNN
# compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# training the CNN on the training set and evaluating it on the Test set
#cnn.fit(x=training_set, validation_data=test_set, epochs=25)


# PART 4: MAKING A SINGLE PREDICTION
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
