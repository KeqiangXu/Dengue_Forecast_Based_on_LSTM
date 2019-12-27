# -*- coding: utf-8 -*-
####################################################################################
#   Implementation of the following paper                                          #
#                                                                                  #
#   Forecast of Dengue Cases in 20 Chinese Cities Based on Deep learning Method    #
#                                                                                  #
####################################################################################
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import tensorflow as tf
import seaborn as sns 
#import pdb
##############LSTM Model Function Definition##################
# Training Function
def train_lstm(sess, savemodelname, train_x, train_y, val_x, val_y, batch_index):
    losses_train = []
    losses_val = []
    epochs = 2000
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

# Test Function
def test_lstm(sess, test_x):
    predict_y = []
    for step in range(len(test_x)):
        pred_ = sess.run(pred, feed_dict={X:[test_x[step]]})
        predict=pred_.reshape((-1))
        predict_y.append(predict[-1])
    return np.array(predict_y)

# Data Reading
def read_train_data(df):
    data=df.iloc[:,1:].values  # Preserve attributes and labels
    y = data[:,-1]
    mean_ = np.mean(y)
    std_ = np.std(y)
    return mean_, std_

# Data Reading
def read_test_data(df):
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
    # Test Set 
    test_begin=0
    data_test=data[test_begin:]
    mean_test=np.mean(data_test,axis=0)
    std_test=np.std(data_test,axis=0)  # Standard Deviation
    normalized_test_data=(data_test-mean_test)/std_test  # Normalized
    test_x=[]
    for i in range(len(normalized_test_data)-time_step+1):
        x=normalized_test_data[i:i+time_step,:input_size]
        test_x.append(x)
    test_x = np.array(test_x)
    return test_x, data[11:,-1],mean_test, std_test

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

is_training=False

##############Import data##################
X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
pred,_=lstm(X)
# Loss Function
loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
train_op=tf.train.AdamOptimizer(lr).minimize(loss)

# Root Directory
filenames = os.listdir('./train')
filenames_2 = os.listdir('./test_2')
cityname = [i.replace('_train.csv','') for i in filenames]


#for file in filenames:
n = 5 
file = filenames[n]
file_2 = filenames_2[n]
f_train = open('./train/'+file)
f_test = open('./test_2/'+file_2)
df_train = pd.read_csv(f_train)
df_test=pd.read_csv(f_test)

mean_, std_ = read_train_data(df_train)
test_x, true_y,mean_test,std_test = read_test_data(df_test)

savemodelname = './models/model_'+file.split('.')[0]+'/'+'model_'+file.split('.')[0]
try:
    os.mkdir('./models/model_'+file.split('.')[0])
except:
    pass

saver = tf.train.Saver()
sess= tf.Session()
model_file = tf.train.latest_checkpoint('./models/model_'+file.split('.')[0]+'/')
saver.restore(sess, model_file)
pred_y = test_lstm(sess, test_x)
pred_y = pred_y*std_+mean_  # Anti-normalization

prediction = []
for i in range(len(pred_y)):
    if pred_y[i] < math.log(2):
        pred_y[i] = 0
    prediction.append(pred_y[i])

##############Draw Figure##################

dates=pd.date_range('2005-12','2019-01',freq='M') # Create date data at monthly intervals
profit = {'date':list(df_test['date'][11:]),'observation':list(true_y),'prediction':list(pred_y)}
profit = DataFrame(profit)
profit.to_csv('./prediction/prediction_'+file.replace('_train','') ,index = False , sep = ',')
profit.set_axis(dates)

fig1 = plt.figure(figsize=(14,6))
sns.set()
#    sns.set_style('white')
sns.set_style('whitegrid')
plt.ylim(-0.5,11,1)
plt.plot(profit['observation'], linestyle=' ', marker='o', color='black',markersize=5) 
plt.plot(profit['prediction'], linestyle='-',linewidth=2.0, color='r',)
plt.title("%s Predict Incidence"%(cityname[n]),fontsize=20)
plt.ylabel('ln case',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.axvline(x='2017', color='r', linewidth=1.5)
#    plt.axvline(x='2018', color='r', linewidth=1.5)
plt.savefig('./result/prediction_'+file.split('.')[0].split('_')[0]+'.png',dpi=300)

##############Model Performance Evaluation##################

def rmse(data):
    error = []
    for i in range(len(data)):
        error.append(math.ceil(math.exp(data['observation'][i])) - math.ceil(math.exp(data['prediction'][i])))
    squared_error = []
    for val in error:
        squared_error.append(val*val)
    return (math.sqrt(sum(squared_error) / len(squared_error)))

def mae(data):
    error = []
    for i in range(len(data)):
        error.append(math.ceil(math.exp(data['observation'][i])) -math.ceil(math.exp(data['prediction'][i])))
    absError = []
    for val in error:
        absError.append(abs(val))
    return(sum(absError) / len(absError))

def RRSE(a,b):       
# a is predict, b is actual
    aa=a.copy(); bb=b.copy()
    if len(aa)!=len(bb):
        print('not same length')
        return np.nan
    cc = []
    for i in range(len(bb)):
        cc.append(np.mean(bb))

    # MSE of predictor
    m1 = 0
    for i in range(len(aa)):
        m1 = m1 + (aa[i]-bb[i])**2 
    #MSE when using the mean as a predictor
    m2 = 0
    for i in range(len(bb)):
        m2 = m2 + (cc[i]-bb[i])**2 
    m3 = math.sqrt(m1/m2)
    return m3

fitted = profit[:133]
forecasted = profit[133:]
forecasted = forecasted.reset_index(drop=True)

fitted_rmse = rmse(fitted)
forecasted_rmse = rmse(forecasted)
fitted_mae = mae(fitted)
forecasted_mae = mae(forecasted)
fitted_rrse = RRSE(list(forecasted['prediction']),list(forecasted['observation']))
forecasted_rrse = RRSE(list(forecasted['prediction']),list(forecasted['observation']))

print('fitted_rmse = %.4f'%fitted_rmse)
print('fitted_mae = %.4f'%fitted_mae)
print('fitted_rrse = %.4f'%fitted_rrse)
print('forecasted_rmse = %.4f'%forecasted_rmse)
print('forecasted_mae = %.4f'%forecasted_mae )
print('forecasted_rrse = %.4f'%forecasted_rrse)
    
