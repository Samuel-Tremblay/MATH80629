import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters  import gaussian
from matplotlib import cm
from skimage.transform import rescale, resize, downscale_local_mean, seam_carve
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#-------------------------------------
#Preprocessing
#-------------------------------------



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



def IdeaPreProcess(image):
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
        if count %1000 == 0:
            print(count)
        train_x.append(preProcessFunction(e[1]).reshape(-1))
    return train_x


train_y = loadLabelFile('train_labels.csv')

img_train = np.load('train_images.npy', allow_pickle=True,encoding='latin1')
print("Train image Loaded")
img_test = np.load('test_images.npy', allow_pickle=True,encoding='latin1')
print("Test image Loaded")


train_x = preProcessData(img_train, IdeaPreProcess) #25 by 25 matrix
test_x = preProcessData(img_test, IdeaPreProcess)#25 by 25 matrix



X_train = np.array(train_x).reshape(np.array(train_x).shape[0], 25, 25, 1)
X_train_aug = np.concatenate(([np.fliplr(x) for x in X_train], X_train))


#VGG16
#model = tf.keras.models.Sequential([
#tf.keras.layers.Conv2D(32, (3, 3), input_shape=(25, 25, 1), padding='same', activation='relu'),
#tf.keras.layers.Conv2D(32, (3, 3), input_shape=(25,25,1),activation='relu', padding='same'),
#tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#tf.keras.layers.Dropout(0.25),
#tf.keras.layers.Conv2D(64, (3, 3), input_shape=(25,25,1),activation='relu', padding='same'),
#tf.keras.layers.Conv2D(64, (3, 3), input_shape=(25,25,1),activation='relu', padding='same'),
#tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#tf.keras.layers.Dropout(0.25),
#tf.keras.layers.Conv2D(128, (3, 3), input_shape=(25,25,1),activation='relu', padding='same'),
#tf.keras.layers.Conv2D(128, (3, 3), input_shape=(25,25,1),activation='relu', padding='same'),
#tf.keras.layers.Conv2D(128, (3, 3), input_shape=(25,25,1),activation='relu', padding='same'),
#tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#tf.keras.layers.Dropout(0.25),
#tf.keras.layers.Conv2D(256, (3, 3), input_shape=(25,25,1),activation='relu', padding='same'),
#tf.keras.layers.Conv2D(256, (3, 3), input_shape=(25,25,1),activation='relu', padding='same'),
#tf.keras.layers.Conv2D(256, (3, 3), input_shape=(25,25,1),activation='relu', padding='same'),
#tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#tf.keras.layers.Dropout(0.25),
#tf.keras.layers.Conv2D(256, (3, 3), input_shape=(25,25,1),activation='relu', padding='same'),
#tf.keras.layers.Conv2D(256, (3, 3), input_shape=(25,25,1),activation='relu', padding='same'),
#tf.keras.layers.Conv2D(256, (3, 3), input_shape=(25,25,1),activation='relu', padding='same'),
#tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
#tf.keras.layers.Dropout(0.25),
#tf.keras.layers.Flatten(),
#tf.keras.layers.Dense(256, activation='relu'),
#tf.keras.layers.Dropout(0.25),
#tf.keras.layers.Dense(len(categories), activation='softmax')])


#AlexNet
#model = Sequential()

#model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=(25,25,1), activation='relu'))
# Max Pooling
#model.add(MaxPooling2D(pool_size=(2,2)))
# 2nd Convolutional Layer
#model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(25,25,1),activation='relu'))
# Max Pooling
#model.add(MaxPooling2D(pool_size=(2,2)))
# 3rd Convolutional Layer
#model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(25,25,1), activation='relu'))
# Passing it to a Fully Connected layer
#model.add(Flatten())
# 1st Fully Connected Layer
#model.add(Dense(256, activation='relu'))
# Add Dropout to prevent overfitting
#model.add(Dropout(0.4))
# 2nd Fully Connected Layer
#model.add(Dense(128,activation='relu'))
# Add Dropout
#model.add(Dropout(0.4))
# Output Layer
#model.add(Dense(len(categories),activation='softmax'))

#Lenet

#model = tf.keras.models.Sequential([
# C1 Convolutional Layer
#tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(25,25,1), padding='same'),
# S2 Pooling Layer
#tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'),
#tf.keras.layers.Dropout(0.25),
# C3 Convolutional Layer
#tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'),
# S4 Pooling Layer
#tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
#tf.keras.layers.Dropout(0.25),
# C5 Fully Connected Convolutional Layer
#tf.keras.layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'),
#Flatten the CNN output so that we can connect it with fully connected layers
#tf.keras.layers.Flatten(),
#tf.keras.layers.Dropout(0.25),
# FC6 Fully Connected Layer
#tf.keras.layers.Dense(84, activation='tanh'),
#Output Layer with softmax activation
#tf.keras.layers.Dense(len(categories), activation='softmax')
#)

# Resnet50

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform


# X - output of the convolutional block, tensor of shape (n_H, n_W, n_C)
def convolutional_block(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. We'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


input_shape = (25, 25, 1)
 #Define the input as a tensor with shape input_shape
X_input = Input(input_shape)
# Zero-Padding
X = ZeroPadding2D((3, 3))(X_input)
# Stage 1
X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', )(X)
X = BatchNormalization(axis=3, name='bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides=(2, 2))(X)
# Stage 2
X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
# Stage 3
X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
# Stage 4
X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
# output layer
X = Flatten()(X)
X = Dropout(0.25)(X)# a verifier
X = Dense(len(categories), activation='softmax', name='fc' + str(len(categories)))(X)
# Create model
model = Model(inputs=X_input, outputs=X, name='ResNet50')

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
