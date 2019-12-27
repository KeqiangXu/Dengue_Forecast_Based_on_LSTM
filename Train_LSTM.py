# -*- coding: utf-8 -*-
####################################################################################
#   Implementation of the following paper                                          #
#                                                                                  #
#   Forecast of Dengue Cases in 20 Chinese Cities Based on Deep learning Method    #
#                                                                                  #
####################################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

##############LSTM Model Function Definition##################

# LSTM Model
def lstm(X):
    # Weight Definition
    weights={
             'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
             'out':tf.Variable(tf.random_normal([rnn_unit,1]))
            }
    # Biases Definition
    biases={
            'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
            'out':tf.Variable(tf.constant(0.1,shape=[1,]))
           }
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    #Transformed into a 2-dimensional matrix, and the calculation results are input into the hidden layer
    input=tf.reshape(X,[-1,input_size])  
    input_rnn=tf.matmul(input,w_in)+b_in
    #Convert to the 3D matrix and input the calculation results to LSTM cell
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])   
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit,forget_bias=1.0, state_is_tuple=True)
    # Add dropout
    if is_training :
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True) # lstm cell stack
    #State Initialization
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

# Training Function
def train_lstm(sess, savemodelname, train_x, train_y, val_x, val_y, batch_index):
    losses_train = []
    losses_val = []
    for i in range(epochs):     #Number of epochs
        loss_epoch = 0.
        loss_val_epoch = 0.
        for step in range(len(batch_index)-1):
            _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            loss_val = sess.run([loss], feed_dict={X:val_x,Y:val_y})
            loss_epoch += loss_
            loss_val_epoch+= loss_val[0]
        losses_val.append(loss_epoch/step)
        losses_train.append(loss_val_epoch/step)
        if (i+1)%100 == 0:
            print("Number of iterations:",i+1, " loss:", loss_epoch/step, "val loss:", loss_val_epoch/step)
            saver.save(sess, savemodelname, global_step=i+1)
    print("The train has finished")  
    return losses_train, losses_val

# Data Reading
def read_data():
    reduction = ['extreme_wind_speed',
                 'mean_wind_speed',
                 'maximum_wind_speed',
                 'mean_temperature_of_air',
                 'mean_minimum_temperature',
                 'minimum_pressure'
                 ]
    for s in reduction:
        del df['%s'%(s)]
    data=df.iloc[:,1:].values  # Preserve attributes and labels
    batch_index=[]
    # Test Set 
    train_begin=0
    train_end=144
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)
    # Normalized
    train_x,train_y=[],[]
    for i in range(len(normalized_train_data)-time_step):
        if i % batch_size==0:
            batch_index.append(i)
        x=normalized_train_data[i:i+time_step,:input_size]
        y=normalized_train_data[i:i+time_step,input_size,np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    
    # Validation Set
    test_begin=144
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)  # Standard Deviation
    normalized_test_data=(data_test-mean)/std  # Standard Deviation
    size=(len(normalized_test_data)+time_step-1)//time_step  
    test_x,test_y=[],[]
    for i in range(size-1):
        x=normalized_test_data[i:i+time_step,:input_size]
        y=normalized_test_data[i:i+time_step,input_size]
        test_x.append(x)
        test_y.append(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:input_size]).tolist())
    test_y.append((normalized_test_data[(i+1)*time_step:,input_size]).tolist())
    test_x = np.array(test_x)
    test_y = np.expand_dims(np.array(test_y), axis=-1)
    test_x = test_x.tolist()
    test_y = test_y.tolist()
    return train_x, train_y, test_x, test_y, batch_index


##############Parameter##################
tf.reset_default_graph()
num_layers = 1     # Number of Hidden Layers
rnn_unit = 64      # Number of Hidden Neurons
input_size = 10    # Input Layer
output_size=1      # Output Layer
lr=0.00001         # Learning Rate
keep_prob = 0.4    # Dropout Rate
batch_size=24      # Batch Size
time_step=12       # Time Step

is_training=True
epochs = 10000     # Number of epochs

##############Import data##################

# Root Directory
filenames = os.listdir('./train')

X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
pred,_=lstm(X)
# Loss Function
loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
train_op=tf.train.AdamOptimizer(lr).minimize(loss)

#for file in filenames:
file = filenames[5] 
f=open('./train/'+file)
df=pd.read_csv(f)     #数据读入
train_x, train_y, val_x, val_y, batch_index = read_data()
savemodelname = './models_2/model_'+file.split('.')[0]+'/'+'model_'+file.split('.')[0]
try:
    os.mkdir('./models_2/model_'+file.split('.')[0])
except:
    pass
sess = tf.Session()
sess.run(tf.global_variables_initializer()) 

# Save Model
saver=tf.train.Saver(tf.global_variables(),max_to_keep=150000)
model_file = tf.train.latest_checkpoint('./models/model_'+file.split('.')[0]+'/')
losses_train, losses_val = train_lstm(sess, savemodelname,train_x, train_y, val_x, val_y, batch_index)

##############Draw Figure##################

plt.figure(figsize=(20,10))
yaxis=[i+1 for i in range(len(losses_train))]
plt.plot(yaxis, losses_train)
plt.plot(yaxis, losses_val)
plt.legend(['losses_train', 'losses_val'])
plt.savefig('loss'+file.split('.')[0]+'.png',dpi=300)


    
