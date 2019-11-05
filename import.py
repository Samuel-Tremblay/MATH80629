import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from os.path import join
import csv

def load_data():
    with open(join('data', 'train_labels.csv'), 'r') as f:
        reader = csv.reader(f)
        train_labels = dict(reader)
    train = np.load(join('data', 'train_images.npy'), encoding='latin1')
    test = np.load(join('data', 'test_images.npy'), encoding='latin1')
    
    for x in train:
        x[0] = train_labels[str(x[0])]

    return train, test

if __name__ == '__main__':
    train, test = load_data()
breakpoint()
