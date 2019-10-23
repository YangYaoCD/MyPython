import numpy as np
import h5py
import matplotlib.pyplot as plt
    
    
def load_dataset():
    train_dataset = h5py.File('C:\\Users\\yangyao\\Desktop\\课件\\deeplearning.ai-master(1)\\deeplearning.ai-master\\COURSE 1 Neural Networks and Deep Learning\\Week 2\\Logistic Regression as a Neural Network\\datasets\\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('C:\\Users\\yangyao\\Desktop\\课件\\deeplearning.ai-master(1)\\deeplearning.ai-master\\COURSE 1 Neural Networks and Deep Learning\\Week 2\\Logistic Regression as a Neural Network\\datasets\\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
