import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D

lines = []

#Read in csv info
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

#Create lists of image directories and steering angles
image_dirs = []
measures = []
correction = 0.25
sign = [0.0, 1.0, -1.0]
for line in lines:
    for index in range(3):
        path = line[index]
        filename = path.split('\\')[-1]
        full_path = './data/IMG/' + filename
        image_dirs.append(full_path)
        measure = float(line[3]) + (sign[index] * correction)
        measures.append(measure)

#Convert them to numpy arrays for keras
X_train = np.array(image_dirs)
y_train = np.array(measures)

#Split data into training and validation groups
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

#Convert image to YUV for the NVidia model
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

#Flip images to remove track left turn bias
def flip_image(image, measure):
    flip = np.random.randint(0,2)
    if flip == 0:
        return image,measure
    elif flip == 1:
        image,measure = cv2.flip(image,1),-measure
        return image,measure

#Translate images horizontally and vertically to help generalize
def random_translate(image, measure):
    dx = 100
    dy = 15
    new_x = dx * (np.random.rand() - 0.5)
    new_y = dy * (np.random.rand() - 0.5)
    measure += new_x * 0.003
    move = np.float32([[1, 0, new_x], [0, 1, new_y]])
    row, col = image.shape[:2]
    image = cv2.warpAffine(image, move, (col, row))
    return image, measure

#Add a shadow to the image, helping with conditions found within the track
def shadow_image(image):
    #SOURCE: https://github.com/naokishibuya/car-behavioral-cloning/blob/master/utils.py

    row,col,c = image.shape
    x1, y1 = col * np.random.rand(), 0
    x2, y2 = col * np.random.rand(), row
    xm, ym = np.mgrid[0:row, 0:col]
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

#Randomly alter the brightness of the image.
def alter_brightness(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #valus for uniform below were by trial and error
    hsv_image[:,:,2] = hsv_image[:,:,2]*np.random.uniform(0.1,1.25)
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image

#Pipeline for augmenting the image
def augment_image(image, measure):
    image, measure = flip_image(image, measure)
    image, measure = random_translate(image, measure)
    image = shadow_image(image)
    image = alter_brightness(image)
    image = rgb2yuv(image)
    return image, measure

#Simple data generator for feeding the network. Takes the original images and also adds in my augmented ones.
def generator(X, y, batch_size=64):
    num_images = len(X)
    while True:
        shuffle(X, y)
        for offset in range(0, num_images, batch_size):
            X_batch = X[offset:offset+batch_size]
            y_batch = y[offset:offset+batch_size]

            images = []
            measurements = []
            for X_dir, y_value in zip(X_batch, y_batch):
                image = cv2.imread(X_dir)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                measurement = float(y_value)

                images.append(image)
                measurements.append(measurement)

                augmented_image, augmented_measure = augment_image(image, measurement)
                images.append(augmented_image)
                measurements.append(augmented_measure)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield sklearn.utils.shuffle(X_train, y_train)

#Model Definition and generator declaration
keep_prob = 0.5

train_generator = generator(X_train, y_train, batch_size=64)
validation_generator = generator(X_valid, y_valid, batch_size=64)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#model.summary()

#Train the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(X_train), validation_data=validation_generator, nb_val_samples = len(X_valid), nb_epoch=7, verbose=1)

#Save the model
model.save('model_simple_with_augmentation.h5')

#Plot the training and validation loss for each epoch
#Source: UDACITY course videos
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('MSE over Epochs')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch number')
plt.legend(['training data', 'validation data'], loc='upper right')
plt.savefig('./Figure.png')

