""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the air pollution   database of pm2.5 in Taiwan
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Dennis Liu
Refference Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""


from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.contrib import rnn

# Import data
from data_generator import data_generator
from networks import model
from config import *


pretrained = None
data_root = '/root/data/pm25_data/datas/sensors/'

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])


# get graph
prediction , loss_op , train_op , accuracy = model(X , Y , 'muti_lstm' , training = True , lr = 0.001)


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# create data iterator of training set and val set
train_iter = data_generator(os.path.join(data_root,'train'),batch_size,sample_feq,sample_skip,sample_times = 10000)
val_iter = data_generator(os.path.join(data_root,'val'),batch_size,sample_feq,sample_skip,sample_times = 1000)

# Start training
with tf.Session() as sess:
    saver = tf.train.Saver()  
    
    # load pretrain model or initial
    if pretrained:
        saver.restore(sess , pretrained)
    else:
        sess.run(init) 
    
    
    
    for epoch in range(training_steps):
        for step , (batch_x, batch_y) in enumerate(train_iter):
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
#                 print (sess.run([prediction, Y], feed_dict={X: batch_x,Y: batch_y}))
        train_iter.current_time = 0
        
        print("Optimization Finished!")
        # Calculate accuracy for 128 mnist test images
        
        val_len = 0
        total_acc = 0
        for i , (batch_x, batch_y) in enumerate(val_iter):
            acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            if not acc > 0 :
                acc = 0
            total_acc += acc
            val_len += 1 
        val_iter.current_time = 0
        print("Epoch : {} , Testing Accuracy: {} , number of samples : {}".format(epoch , total_acc / val_len , val_len))
        save_path = saver.save(sess, "models/lstm_{}.ckpt".format(epoch))
        print ("Model saved in file: ", save_path)