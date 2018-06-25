import tensorflow as tf
from tensorflow.contrib import rnn

from config import *

def basic_lstm_graph(x):
    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    lstm_cells = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cells, x, dtype=tf.float32)
    
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def multi_lstm_graph(x):
    
    lstm_cells = [rnn.BasicLSTMCell(num_hidden, forget_bias=1.0) for i in range(3)]
    lstm_multi = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)  
    
    # Get lstm cell output
    outputs, states = tf.nn.dynamic_rnn(lstm_multi, x, dtype=tf.float32)  
    
    outputs = tf.contrib.layers.flatten(outputs)
    outputs = tf.layers.dense(outputs, 256, activation=tf.nn.relu)
    outputs = tf.layers.dense(outputs, num_classes)
    
    return outputs



def model(X , Y = None , network = 'basic_lstm' , training = True , lr = 0.001):
    if network == 'basic_lstm':
        logits = basic_lstm_graph(X)
    elif network == 'muti_lstm':
        logits = multi_lstm_graph(X)
        
    prediction = tf.nn.sigmoid(logits)
    
    if training :
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss_op)
        # Evaluate model (with test logits, for dropout to be disabled)
        accuracy = tf.reduce_mean(tf.maximum(1 - tf.abs(prediction - Y) / Y , 0) ) 
        return prediction , loss_op , train_op , accuracy
    else:
        return prediction
    

def test_graph():
    
    # test basic_lstm in training mode
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    
    model(X , Y , 'basic_lstm' , training = True , lr = 0.001)
    tf.reset_default_graph()
    
    # test muti_lstm in training mode
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    model(X , Y , 'muti_lstm' , training = True , lr = 0.001)
    tf.reset_default_graph()
    
    # test basic_lstm in testing mode
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    model(X , Y , 'basic_lstm' , training = False)
    tf.reset_default_graph()
    
    # test muti_lstm in testing mode
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    model(X , Y , 'muti_lstm' , training = False)
    tf.reset_default_graph()
    
    

if __name__ == '__main__':
    test_graph()
    
    
    