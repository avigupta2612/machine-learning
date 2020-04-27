import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense,Flatten,Dropout,BatchNormalization
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
x_train, x_test = x_train / 255.0, x_test / 255.0
model=tf.keras.models.Sequential()
model.add(Conv2D(32,3,activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(64,3,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=5,
          steps_per_epoch=512,
          batch_size=64,
          verbose=1)
          
test_loss, test_acc = model.evaluate(x_test,y_test)
print(test_acc)
import numpy as np
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [28,28])
    #image = image.reshape((28, 28, 1))
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


p='C:\\Users\\Avi\\Desktop\\Python Progs\\image1.jpg'
img=load_and_preprocess_image(p)
img=tf.image.rgb_to_grayscale(
    img,
    name=None
)
print(img.shape)


img=tf.reshape(img,[1,28,28,1])
predict=model.predict(img)
label=predict.argmax(axis=-1)
#print(categories[label[0]])
print(label[0])

