#
# import numpy as np
# import copy
#
# class NeuralNetwork:
#     def __init__(self,input_size,hidden_size,output_size):
#         self.input_size=input_size
#         self.hidden_size=hidden_size
#         self.output_size=output_size
#
#         # Împărțim la sqrt(dimensiune) pentru a normaliza varianța activărilor
#         # (Xavier/He initialization) și a preveni vanishing/exploding gradients
#         self.W1=np.random.randn(hidden_size,input_size)/np.sqrt(input_size)
#         self.b1=np.random.randn(hidden_size,1)/np.sqrt(hidden_size)
#         self.W2=np.random.randn(output_size,hidden_size)/np.sqrt(hidden_size)
#         self.b2=np.random.randn(output_size,1)/np.sqrt(hidden_size)
#
#     def _relu(self,Z,derivative=False):
#         if derivative:
#             return (Z>0).astype(float)
#         return np.maximum(0,Z)
#
#     def _softmax(self,Z):
#         exp_Z=np.exp(Z-np.max(Z)) #scade valoarea maxima pt a nu provoca overflow
#         return exp_Z/exp_Z.sum(axis=0,keepdims=True)
#
#     def _cross_entropy(self,y_pred,y_true):
#         return -np.log(y_pred[y_true]+1e-8)
#
#     def forward(self,x):
#         self.Z1=np.dot(self.W1,x)+self.b1
#         self.H=self._relu(self.Z1)
#         self.Z2=np.dot(self.W2,self.H)+self.b2
#         self.output=self._softmax(self.Z2)
#         return self.output
#
#     def backward(self,x,y,output):
#         y_one_hot=np.zeros(self.output_size)
#         y_one_hot[y]=1
#
#         print(output.shape)
#         print(y_one_hot.shape)
#         dZ2=output-y_one_hot
#         dW2=np.outer(dZ2,self.H)
#         db2=dZ2.reshape(-1,1)
#
#         dH=np.dot(self.W2.T,dZ2)
#         dZ1=dH*self._relu(self.Z1,derivative=True)
#         dW1=np.outer(dZ1,x)
#         db1=dZ1.reshape(-1,1)
#
#         return {'dW1':dW1,'db1':db1,'dW2':dW2,'db2':db2}
#
#     def update_parameters(self,grads,learning_rate):
#         self.W1-=learning_rate*grads['dW1']
#         self.b1-=learning_rate*grads['db1']
#         self.W2-=learning_rate*grads['dW2']
#         self.b2-=learning_rate*grads['db2']
#
#     def predict(self,x):
#         output=self.forward(x)
#         return np.argmax(output)
#
#     def get_parameters(self):
#         return {
#             'W1':self.W1.copy(),
#             'b1':self.b1.copy(),
#             'W2':self.W2.copy(),
#             'b2':self.b2.copy(),
#             'input_size':self.input_size,
#             'hidden_size':self.hidden_size,
#             'output_size':self.output_size
#         }
#
#     def set_parameters(self,params):
#         self.W1=params['W1'].copy()
#         self.b1=params['b1'].copy()
#         self.W2=params['W2'].copy()
#         self.b2=params['b2'].copy()
#         self.input_size=params['input_size']
#         self.hidden_size=params['hidden_size']
#         self.output_size=params['output_size']
#
import numpy as np
import copy


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)
        self.b1 = np.random.randn(hidden_size, 1) / np.sqrt(hidden_size)
        self.W2 = np.random.randn(output_size, hidden_size) / np.sqrt(hidden_size)
        self.b2 = np.random.randn(output_size, 1) / np.sqrt(hidden_size)

    def _relu(self, Z, derivative=False):
        if derivative:
            return (Z > 0).astype(float)
        return np.maximum(0, Z)

    def _softmax(self, Z):
        # Z trebuie să fie un vector 1D (output_size,)
        exp_Z = np.exp(Z - np.max(Z))
        return exp_Z / exp_Z.sum()

    def _cross_entropy(self, y_pred, y_true):
        return -np.log(y_pred[y_true] + 1e-8)

    def forward(self, x):
        # Asigură-te că x este column vector (input_size, 1)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        self.Z1 = np.dot(self.W1, x) + self.b1
        self.H = self._relu(self.Z1)
        self.Z2 = np.dot(self.W2, self.H) + self.b2

        # Flatten Z2 pentru softmax - returnează vector 1D (output_size,)
        self.output = self._softmax(self.Z2.flatten())
        return self.output

    def backward(self, x, y, output):
        y_one_hot = np.zeros(self.output_size)
        y_one_hot[y] = 1

        # output este acum vector 1D, la fel ca y_one_hot
        dZ2 = output - y_one_hot

        # Reshape pentru operații matriciale
        dZ2 = dZ2.reshape(-1, 1)
        dW2 = np.dot(dZ2, self.H.T)
        db2 = dZ2

        dH = np.dot(self.W2.T, dZ2)
        dZ1 = dH * self._relu(self.Z1, derivative=True)

        # Asigură-te că x este column vector
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        dW1 = np.dot(dZ1, x.T)
        db1 = dZ1

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def update_parameters(self, grads, learning_rate):
        self.W1 -= learning_rate * grads['dW1']
        self.b1 -= learning_rate * grads['db1']
        self.W2 -= learning_rate * grads['dW2']
        self.b2 -= learning_rate * grads['db2']

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output)

    def get_parameters(self):
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }

    def set_parameters(self, params):
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.output_size = params['output_size']