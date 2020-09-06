import random
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2
import time

import tensorflow as tf
from tensorflow.contrib.layers import flatten

from sklearn.utils import shuffle

def pre_process(X, channels=1):
    
    X_out = X
    
    # grayscale
    if channels == 1: 
        X_out = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in X_out]).reshape((-1, 32, 32, 1))
    
    # normalization
    X_out = (X_out - 128)/128
    
    return X_out

def LeNet(x, params):    
    
    Input_channels = params['Input_channels']
    n_classes = params['n_classes']

    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x Input_channels. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, Input_channels, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = n_classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

def evaluate(X_data, y_data, params):

    BATCH_SIZE = params['BATCH_SIZE']
    accuracy_operation = params['accuracy_operation']

    x = params['x']
    y = params['y']

    num_examples = len(X_data)

    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples
    
def Compile_Model(params):
    
    Input_channels = params['Input_channels']
    n_classes = params['n_classes']
    rate = params['rate']

    # define place holders
    x = tf.placeholder(tf.float32, (None, 32, 32, Input_channels))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)
    params['x'] = x
    params['y'] = y

    # setup model
    logits = LeNet(x, params)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)

    # model parameters
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    params['training_operation'] = training_operation # note: for training use
    params['accuracy_operation'] = accuracy_operation # note: for evaluate() use


def Train_and_Test_Model(params, Data, train_validation=False):

    EPOCHS = params['EPOCHS']
    BATCH_SIZE = params['BATCH_SIZE']
    training_operation = params['training_operation']

    x = params['x']
    y = params['y']

    X_train = Data['X_train']
    y_train = Data['y_train']
    X_valid = Data['X_valid']
    y_valid = Data['y_valid']
    X_test = Data['X_test']
    y_test = Data['y_test']

    saver = tf.train.Saver()

    valid_name = 'Validation Accuracy'
    if train_validation:
        valid_name = 'Training-validation Accuracy'

    # Train model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        
        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
            train_accuracy = evaluate(X_train, y_train, params)
            validation_accuracy = evaluate(X_valid, y_valid, params)
            testing_accuracy = evaluate(X_test, y_test, params)
            
            if (i + 1) % 5 == 0:
                print("EPOCH {} ...".format(i+1))
                print("Training Accuracy = {:.3f}".format(train_accuracy))
                print("{0} = {1:.3f}".format(valid_name, validation_accuracy))
                print("Testing Accuracy = {:.3f}".format(testing_accuracy))
                print()
            
        saver.save(sess, './lenet_appendix')
        print("Model saved")



