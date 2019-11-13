import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters  import gaussian
from matplotlib import cm
from skimage.transform import rescale, resize, downscale_local_mean, seam_carve
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

# %% md

#-------------------------------------
#Preprocessing
#-------------------------------------


# %%

# General utils

# Create an array of number form 0 to 1
# where the gap between to consecutive number is 0.1
def creatBins():
    return np.arange(0, 1, 0.01)


# Create and histogram givens bin and an input vector
def createHistogram(normalizeVector, bins):
    return np.histogram(normalizeVector, bins=bins)[0]


# This gives a normalize vector of frequency of pixel. This is a nomalize projection
def getFrequencyRowVector(image):
    result = np.zeros(100)
    for x in range(len(image)):
        result[x] = sum(image[x])
    norm = np.linalg.norm(result)
    if norm == 0:
        return result
    return result / norm


def getFrequencyColumnVector(image):
    transpose = np.transpose(image)
    result = np.zeros(100)
    for x in range(len(transpose)):
        result[x] = sum(transpose[x])
    norm = np.linalg.norm(result)
    if norm == 0:
        return result
    return result / norm


# This gives a vector of frequency of pixel.  This is a projection

def getFrequencyRowVectorNotNormalize(image):
    result = np.zeros(100)
    for x in range(len(image)):
        result[x] = sum(image[x])
    return result


def getFrequencyColumnVectorNotNormalize(image):
    transpose = np.transpose(image)
    result = np.zeros(100)
    for x in range(len(transpose)):
        result[x] = sum(transpose[x])
    return result


def transformToBinaryImageWithTreshold(image, treshold):
    newImage = []

    for x in range(len(list(image))):
        if image[x] < treshold:
            newImage.append(0.0)
        else:
            newImage.append(1.0)

    return np.array(newImage).reshape(100,100)

# Note that this utils was taken from stackoverflow
# https://stackoverflow.com/questions/40824245/how-to-crop-image-based-on-binary-mask/40826140
# This is a slightly modification on the proposed solution.
# This crop the image.
def crop_image(img):
    mask = img > 0
    return img[np.ix_(mask.any(1), mask.any(0))]


# Note that this utils was taken from stackoverflow
# https://stackoverflow.com/questions/10871220/making-a-matrix-square-and-padding-it-with-desired-value-in-numpy
# This automatically pads the image vertically and horizontally with 0.
# The 0 are inserted after the or bellow the image.
# The final square has the size of the biggest dimention.
# Since it is algorithm based I decided to use without modification and add the reference.
def squarify(M, val):
    (a, b) = M.shape
    if a > b:
        padding = ((0, 0), (0, a - b))
    else:
        padding = ((0, b - a), (0, 0))
    return np.pad(M, padding, mode='constant', constant_values=val)

# %% md

def preProcess(img):
    image = img.reshape(100,100)
    res =  ndi.binary_fill_holes(image) #Fill the holes in binary objects
    b = morphology.remove_small_objects(res, 90,10)#Was 100: Remove continguous holes smaller than the specified size inorder to remove the noise
    result = np.zeros(len(img))     #On cherche le bruit (=1) on le rend egale a 0
    b_array = b.reshape(-1)
    for x  in range(len(b_array)): #Ici on ne prend que les 1
        if b_array[x] == True:
            result[x] = img[x]
    return result.reshape(100,100) #a la fin on aura une matrice (image clean sans noise) de 100*100

#%%


def SeventhIdeaPreProcess(image):
    preprocess = preProcess(image)
    preProccessed = transformToBinaryImageWithTreshold(preprocess.reshape(-1), 50)
    row = getFrequencyRowVectorNotNormalize(preProccessed)
    col = getFrequencyColumnVectorNotNormalize(preProccessed)
    trimmed_row = np.trim_zeros(row, 'f')
    trimmed_col = np.trim_zeros(col, 'f')

    full_row = np.concatenate((trimmed_row, np.zeros(100 - len(trimmed_row))), axis=None)
    full_col = np.concatenate((trimmed_col, np.zeros(100 - len(trimmed_col))), axis=None)
    return np.concatenate((full_row, full_col), axis=None)



def NineIdeaPreProcess(image):
    preprocess = preProcess(image)
    preProccessed = transformToBinaryImageWithTreshold(preprocess.reshape(-1), 100).reshape(100, 100)

    if sum(preProccessed.reshape(-1)) == 0:
        return np.zeros((25, 25))
    out = resize(squarify(crop_image(preProccessed), 0), (25, 25))

    return out


categories = [
    'sink',
    'pear',
    'moustache',
    'nose',
    'skateboard',
    'penguin',
    'peanut',
    'skull',
    'panda',
    'paintbrush',
    'nail',
    'apple',
    'rifle',
    'mug',
    'sailboat',
    'pineapple',
    'spoon',
    'rabbit',
    'shovel',
    'rollerskates',
    'screwdriver',
    'scorpion',
    'rhinoceros',
    'pool',
    'octagon',
    'pillow',
    'parrot',
    'squiggle',
    'mouth',
    'empty',
    'pencil'
]


def saveSubmissionFile(filename, content):
    count = 0
    for c in content:
        print(str(count)+ ',' + categories[c]+'')
        count = count + 1

def loadLabelFile(filename):
    data = []
    f = open(filename, 'r')
    count = 0
    for line in f:
        if count != 0:
            data.append(categories.index(line.split(',')[1][0:-1]))
        count = count +1
    f.close();
    return data


def preProcessData(rawData, preProcessFunction): # bich te5ou le deuxieme vecteur mta3 les coordonnees (shape 10000)
    train_x = []
    count = 0
    for e in rawData:
        count=count +1
        if count %1000 == 0: #(condition if zeyda)
            print(count)
        train_x.append(preProcessFunction(e[1]).reshape(-1))
    return train_x


train_y = loadLabelFile('train_labels.csv')

img_train = np.load('train_images.npy', allow_pickle=True,encoding='latin1')
print("Train image Loaded")
img_test = np.load('test_images.npy', allow_pickle=True,encoding='latin1')
print("Test image Loaded")

#%%

train_x = preProcessData(img_train, NineIdeaPreProcess) #25 by 25 matrix
test_x = preProcessData(img_test, NineIdeaPreProcess)#25 by 25 matrix

train_x_flat = preProcessData(img_train, SeventhIdeaPreProcess) #200 input vector
test_x_flat = preProcessData(img_test, SeventhIdeaPreProcess) #200 input vector


print(train_x)


X_train = np.array(train_x).reshape(np.array(train_x).shape[0], 25, 25, 1)
X_train_aug = np.concatenate(([np.fliplr(x) for x in X_train], X_train))

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(25, 25, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(25, 25, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(25, 25, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(25, 25, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25), #to prevent overfitting

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(len(categories), activation=tf.nn.softmax) #output layers
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_aug, np.concatenate((np.array(train_y), np.array(train_y))), validation_split=0.3,
                    epochs=75, shuffle=True)

# TAKEN FROM https://keras.io/visualization/
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
