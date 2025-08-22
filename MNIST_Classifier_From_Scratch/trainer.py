

import numpy as np
import pickle
import os

class NeuralNetworkTrainer:
    def __init__(self,model):
        self.model=model
        self.training_history={
            'loss':[],
            'accuracy':[],
            'test_accuracy':[]
        }

    def learning_rate_schedule(self,base_rate,iteration,total_iterations):
        """Learning rate schedule with step decay"""
        return base_rate*(0.1**(iteration//(total_iterations//3)))

    def train(self,x_train,y_train,x_test,y_test,num_iterations=1000,learning_rate=0.01,eval_interval=1000):
        for i in range(num_iterations):
            idx=np.random.randint(len(x_train))
            x=x_train[idx]
            y=y_train[idx]

            current_lr=self.learning_rate_schedule(learning_rate,i,num_iterations)
            output=self.model.forward(x)
            grads=self.model.backward(x,y,output)
            self.model.update_parameters(grads,current_lr)

            if i%eval_interval==0 or i==num_iterations-1:
                train_acc=self.evaluate(x_train,y_train,subset_size=1000)
                test_acc=self.evaluate(x_test,y_test,subset_size=1000)
                self.training_history['loss'].append((i,self._compute_loss(x_train,y_train,subset_size=1000)))
                self.training_history['accuracy'].append((i,train_acc))
                self.training_history['test_accuracy'].append((i,test_acc))
                print(f"Iteration {i}: Train Acc={train_acc:.4f},Test Acc={test_acc:.4f}, LR={current_lr:.6f}")

    def evaluate(self,x_data,y_data,subset_size=None):
        if subset_size and len(x_data) > subset_size:
            indices=np.random.choice(len(x_data),subset_size,replace=False)
            x_subset=x_data[indices]
            y_subset=y_data[indices]
        else:
            x_subset=x_data
            y_subset=y_data
        correct=0
        for i in range(len(x_subset)):
            prediction=self.model.predict(x_subset[i])
            if prediction==y_subset[i]:
                correct+=1
        return correct/len(x_subset)

    def _compute_loss(self,x_data,y_data,subset_size=1000):
        indices=np.random.choice(len(x_data),min(subset_size,len(x_data)),replace=False)
        total_loss=0

        for idx in indices:
            x=x_data[idx]
            y=y_data[idx]
            output=self.model.forward(x)
            total_loss+=-np.log(output[y]+1e-8)
        return total_loss/len(indices)

    def save_model(self,filepath):
        """Save model parameters to file"""
        model_data=self.model.get_parameters()
        with open(filepath,'wb') as f:
            pickle.dump(model_data,f)

    def load_model(self,filepath):
        """Load model parameters from file"""
        if os.path.exists(filepath):
            with open(filepath,'rb') as f:
                model_data=pickle.load(f)
            self.model.set_parameters(model_data)
            return True
        return False


