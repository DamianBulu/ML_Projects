
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.x_train=None
        self.y_train=None
        self.x_test=None
        self.y_test=None

    def load_mnist_data(self,test_size=0.2,random_state=42):
        """Load MNIST data using sklearn's fetch_openml"""
        mnist=fetch_openml('mnist_784',version=1,as_frame=False,parser='auto')
        x=mnist.data.astype(np.float32)/255.0
        y=mnist.target.astype(np.int32)

        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(x,y,test_size=test_size,random_state=random_state)

        return self.x_train,self.x_test,self.y_train,self.y_test
    def get_data_stats(self):
        """Get basic statistics about the loaded data"""
        stats={
            'train_samples':len(self.x_train) if self.x_train is not None else 0,
            'test_samples':len(self.x_test) if self.x_test is not None else 0,
            'input_dim':self.x_train.shape[1] if self.x_train is not None else 0,
            'num_classes':len(np.unique(self.y_train)) if  self.y_train is not None else 0
        }
        return stats