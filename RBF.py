
# coding: utf-8
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import sys
import time

COLS           = ['A']
TARGET_COL     = 'Target'

# Split into Train and Test
def split(df, pct):
    df_train, df_test = train_test_split(df, test_size=pct)

    train_x = df_train[COLS]
    train_y = pd.DataFrame(df_train[TARGET_COL])
    test_x = df_test[COLS]
    test_y = df_test[TARGET_COL]
    return train_x, train_y, test_x, test_y

# Each node has a Center, randomly assigned. Distances are computed to each Center
def get_centers(X, method, nodes):
    if method == 'km':
        return KMeans(n_clusters=nodes).fit(X).cluster_centers_
    else:
        return np.array(X.sample(nodes)['A']).reshape(nodes,1)

# Load the data
def process(df,parms):
    out = open('/home/tom/data/rbf_timings.csv', 'a')
    
    SIGMA          = parms[0]
    CLUSTER_METHOD = parms[1]
    NUM_NODES      = parms[2]
    TEST_PCT       = parms[3]
    
    #rec = '{}{}{}{}{}{}{}'.format('Sigma,','Cluster,','Nodes,','TestPct,','Time1,','Time2','\n')
    #out.write(rec)
    
    train_x, train_y, test_x, test_y = split(df, TEST_PCT)
    centers = get_centers(train_x, CLUSTER_METHOD, NUM_NODES)
    
# Set up the design matrix
    c   = tf.placeholder("float", shape=[train_x.shape[1]])
    x   = tf.placeholder("float", shape=[None,train_x.shape[1]])
    y_  = tf.placeholder("float", shape=[None,train_y.shape[1]])
    DM  = tf.placeholder("float", shape=[None,NUM_NODES])
# "rbf" is the radial basis function. Each "x" is processed by each node, so for 50
# samples and a 3-node network, you'd produce a 50x3 array
# Each node will have a different "c" Center but otherwise the same
    rbf = tf.exp(-tf.div(tf.pow(tf.sub(x, c),2), tf.pow(SIGMA,2)))

# Loop through the nodes. Use a different "c" each time and accumulate results in rbf_array
# The array has an initial "zeros" just to get the shape right and allow "insert"
    rbf_array = np.zeros(shape=[len(train_x),1])

    for i in range(NUM_NODES):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            tom = sess.run(rbf, feed_dict={x: train_x, c: centers[i]})
        rbf_array = np.insert(rbf_array, i+1, tom.flatten(), axis=1)
# Now you can delete that placeholder column of zeros
    rbf_array = np.delete(rbf_array,0, axis=1)
    
# Optimize the weights with a series of matrix manipulations
    step1 = tf.matmul(DM, DM, transpose_a=True)
    step2 = tf.matrix_inverse(step1)
    step3 = tf.matmul(step2,DM, transpose_b=True)
    weights = tf.matmul(step3, y_)

# Run the optimization job
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        opt_weights = sess.run(weights, feed_dict={DM: rbf_array, y_: train_y})

# Generate fitted values
# Now that the weights are optimized, run the test batch through and see what you get
# First, generate a new Design Matrix, which is the test_x values run through the rbf
    rbf_array = np.zeros(shape=[len(test_x),1])

    for i in range(NUM_NODES):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            tom = sess.run(rbf, feed_dict={x: test_x, c: centers[i]})
        rbf_array = np.insert(rbf_array, i+1, tom.flatten(), axis=1)
    rbf_array = np.delete(rbf_array,0, axis=1)

# Now multiply the DM by the weights for the fitted values

    t1 = time.time()
    final = tf.matmul(DM, weights)
    with tf.Session() as sess:
        fitted_y = sess.run(final, feed_dict={DM: rbf_array, weights: opt_weights})

    t2 = time.time()
    rec = str(SIGMA)+','+CLUSTER_METHOD+','+str(NUM_NODES)+','+str(TEST_PCT)+','+str(t2-t1)+'\n'
    out.write(rec)
    return(np.sqrt(np.mean((fitted_y-test_y.values.reshape(len(fitted_y),1) )**2)))