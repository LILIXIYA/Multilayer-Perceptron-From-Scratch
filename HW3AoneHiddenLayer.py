# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 22:45:12 2019

@author: lxs
Now we have relu in the output and sigmond in the hidden layers.
We use MSE as loss function.

"""
import numpy as np
import xlrd

from sklearn.metrics import confusion_matrix  
import matplotlib.pyplot as plt

#%% Read in the data and standardization
def data_aqc(filename, sheet, start, end):
    # This function is for data acquisition from an Excel file. And for the 
    # inputs, sheet is the sheet number (not index), start and end are the 
    # start and end row indice in the table (not in Python's counting system).
    # The output is the data (X: column_0, Y: column_1), which will be used in
    # future regression algorithm.
    number = end - start + 1 # the number of data ponits from the Excel table
    data = xlrd.open_workbook(filename)
    table = data.sheets()[sheet]
    X = np.zeros((number,2))
    Y = np.zeros(number)
    
    for i in range(0, number):
        X[i,0] = table.cell(start + i -1, 0).value
        X[i,1] = table.cell(start + i -1, 1).value
        Y[i] = table.cell(start + i -1, 2).value
        # in Python the first row index is 0, while in Excel it's 1
    
    return X, Y

def Standardization(data):
    # This function is to perform a Standardization on the input data set and
    # returns the resulting data set.
    data_s = np.sum(data, axis=1)
    data_t = np.zeros((len(data),2))
    m = np.mean(data_s)
    s = np.std(data_s)
    for i in range(len(data)):
        data_t[i,0] = (data[i,0] - m)/s
        data_t[i,1] = (data[i,1] - m)/s
    return data_t

#%% Avtivation functions we are going to use
def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def relu(x):
    x[x<0] = 0
    return x

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoidDerivative(x):
    return x * (1.0 - x)

#%% Build the Neuron graph
class NeuronlGraph:
    
    def __init__(self, X, Y, INI, Num_inputLayer=2, Num_hiddenLayer=1, \
                Size_hiddenLayer=[10,10], Num_outputLayer=1, lr = 0.00001):
        # Stop thresold
        self.thr = 1e-1
        # Learning rate
        self.lr = lr
        # The input 
        self.input = X
        # The out put
        self.y = Y
        # Loss
        self.loss = []
        # Initialization
        self.Initialization(INI, Num_inputLayer, Size_hiddenLayer, Num_outputLayer)
      
    def Initialization(self, INI, Num_inputLayer, Size_hiddenLayer, Num_outputLayer):
        if INI == 'Normal':
            # Initialize the weights
            self.w_1 = np.random.random_sample((Num_inputLayer, Size_hiddenLayer[0]))
            self.w_3 = np.random.random_sample((Size_hiddenLayer[1], Num_outputLayer))
            # Initialize the bias
            self.b_1 = np.random.random_sample(Size_hiddenLayer[0])
            self.b_3 = np.random.random_sample(Num_outputLayer)
        
        if INI == 'Gaussian':
            
            sigma = 1
            # Initialize the weights
            self.w_1 = np.random.normal(loc=0, scale=sigma, size=(Num_inputLayer, Size_hiddenLayer[0]))

            self.w_3 = np.random.normal(loc=0, scale=sigma, size=(Size_hiddenLayer[1], Num_outputLayer))
            # Initialize the bias
            self.b_1 = np.random.normal(loc=0, scale=sigma, size=Size_hiddenLayer[0])
          
            self.b_3 = np.random.normal(loc=0, scale=sigma, size=Num_outputLayer)
            
            
    def feedforward(self):
        self.h1 = relu(np.dot(self.input, self.w_1) + self.b_1)

      
        self.output = sigmoid(np.dot(self.h1, self.w_3) + self.b_3)
       
    def backprop(self,loss_type):
        # derivative of weights
        if loss_type == 'MSE':
            d_temp1 = 2*(self.y - self.output) * sigmoidDerivative(self.output)
        
        d_w3 = np.dot(self.h1.T, d_temp1)
        
        d_temp3 = np.dot(d_temp1, self.w_3.T) * reluDerivative(self.h1)
        d_w1 = np.dot(self.input.T, d_temp3)
   
        # update weights
        self.w_1 += d_w1 * self.lr

        self.w_3 += d_w3 * self.lr
        
        # derivate of bias
        d_b3 = np.sum(d_temp1, axis=0)
     
        d_b1 = np.sum(d_temp3, axis=0)  
        # update bias
        self.b_1 += d_b1 * self.lr
   
        self.b_3 += d_b3 * self.lr
        
    def calLoss(self, loss_type):
        if loss_type == 'MSE':
            loss = np.sum(np.square(self.y - self.output))
            self.loss.append(loss)
       
        #return loss
        
    def train(self, loss_type, N_epoch=1000):
        for i in range(N_epoch):
            self.feedforward()
            self.backprop(loss_type)
            self.calLoss(loss_type)
            # Decide when the training stops
            if i > 2:
                if self.loss[-1] < self.thr * self.loss[-2]:
                    print('Training stops!')
                    break
            
    def predict(self, X):
        self.input = X
        self.feedforward()
        
        pred = self.output
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        
        return pred
    
#%% Get the data set
filename = 'E:\Tue\jaar2Q1\DataMining\HW3Atrain.xlsx'
filenameV = 'E:\Tue\jaar2Q1\DataMining\HW3Avalidate.xlsx'

sheet = 0
X, Y = data_aqc(filename, sheet, 2, 411)
X_validation, Y_validation = data_aqc(filenameV, sheet, 2, 83)
X = Standardization(X)
X_validation = Standardization(X_validation)

#%%Training
NN = NeuronlGraph(X, Y.reshape((len(Y),1)),INI='Gaussian')
NN.train(loss_type = 'MSE')
loss = NN.loss
np.save('loss',loss)
#%% Predict
pre = NN.predict(X_validation)

#%% Confusion matrix
cm = confusion_matrix(Y_validation, pre)
acc = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[1,1] + cm[0,1] + cm[1,0])

#%% Plot
plt.matshow(cm,cmap=plt.cm.Greens)

class_names = ['0', '1']

for x in range(len(cm)):
    for y in range(len(cm)):
        plt.annotate(cm[x,y],xy=(x,y),horizontalalignment='center',verticalalignment='center')