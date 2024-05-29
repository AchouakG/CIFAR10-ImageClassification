import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
#testing and training tuples

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
#this load_data function returns the testing and training data in this format
#the image_classifier.model are arrays of pixels and the training labels are like 'cat', 'plane'...
training_images, testing_images = training_images / 255, testing_images / 255
# the image_classifier.model fed to the network are not with a high resolution
class_names = ['Plane', 'Car', 'Bird', ' Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
# here i am using a 4x4 grid to visualize the dataset
# 4x4 grid with each iteration we are choosing a one of these places in the grid to place the next image

#here is shows the image_classifier.model of each one of those ????
# for i in range(16):
#     plt.subplot(4,4,i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])
# plt.show()
#888888888888888888888
#we are getting the label of the particular image (the number) then we are passing it to as the index for the class list---> if the image label is 3 we're going to get cat
training_images=training_images[:200000]
training_labels=training_labels[:200000]
testing_images= testing_images[:4000]
testing_labels=testing_labels[:4000]

# used to build or train the model for the neural network
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.Flatten()) #this flatten layer takes a multidimensional array and turns it into one-dimensional array
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))
#
# loss, accuracy =model.evaluate(testing_images, testing_labels)
#
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")
#
# model.save('image_classifier.keras')
# i got
# Loss: 0.8822863698005676
# Accuracy: 0.7045000195503235 which is pretty good



model = tf.keras.models.load_model('image_classifier.keras') # now it can be used to classify image_classifier.model from same category
img = cv.imread('cat.jpg')
img= cv.cvtColor(img, cv.COLOR_BGR2RGB)#converting BGR to RGB
# img = cv.resize(img, (32, 32))  # Resize to 32x32 pixels
# img = img / 255 # Normalize the image
plt.imshow(img,cmap=plt.cm.binary)
prediction = model.predict(np.array([img])/255)#we pass an img in a numpy array

index= np.argmax(prediction)
print(f'Prediction is {class_names[index]}')
plt.show()




