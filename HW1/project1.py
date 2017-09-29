import numpy as np
from numpy import genfromtxt

train_set = genfromtxt('train1.csv', delimiter=',')
X = train_set[0:,0:5] #extracts just input varibles into seperate matrix
weightMatrixWidth_W = X.shape[1] #number of input variables needed for making wieghts matrix
X = np.insert(X, 5, 1, axis=1) #appends a 1 to the end of each row so that it can be used to calculate the bias wieght
Y = train_set[0:,5:]  #extracts just output varibles into seperate matrix
weightMatrixWidth_V = Y.shape[1]

numHidNodes = 3 #intial number of hidden nodes will be 2
eta = 0.01 #graidient decent constant

np.random.seed(1)
#creates a random weights matrix between nodes input and hidden. Size is rows= is the length of the number of input varible vector + 1 to account for bias, colums= number of hidden nodes
W = np.random.random((weightMatrixWidth_W+1,numHidNodes)) #weights from input X to hidden layer
V = np.random.random((numHidNodes+1,weightMatrixWidth_V)) #weights from hidden layer to output layer

#define the sigmoid function. If the derivative is needed the it returns x*(1-x) which is the derivative of the sigmoid function x=1/(1+np.exp(-x))
def sigmoid(x, derivative=False):
    if(derivative == True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))

for i in range(150000):
    H = sigmoid(np.dot(X,W)) #gives me values for h1 and h2 hiden nodes
    H = np.insert(H, numHidNodes, 1, axis=1) #adds additional 1 for bias variable on output
    y = sigmoid(np.dot(H,V))

    #start back propagation

    error_output = Y - y #computes the diffence in desired versus output from the neural network

    if (i%10000) == 0:
        print("Error: " + str(np.mean(np.abs(error_output))))

    delta_output = error_output * sigmoid(y, derivative=True) #computes the delta at the output layer k

    error_hidden = np.dot(delta_output,V[0:numHidNodes,0:].T) #computes the error or the hidden layer

    delta_hidden = error_hidden * sigmoid(H[0:,0:numHidNodes], derivative=True) #computes the delta of the hidden layer j

    V += eta * np.dot(H.T,delta_output) #update all the weights from the hidden layer to the output layer
    W += eta * np.dot(X.T,delta_hidden)

print(y)
