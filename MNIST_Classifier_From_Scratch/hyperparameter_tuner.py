

import numpy as np
import pickle
import os
from model import NeuralNetwork
from trainer import NeuralNetworkTrainer

class HyperparameterTuner:
    def __init__(self,x_train,y_train,x_test,y_test):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.best_model=None
        self.best_score=0
        self.best_params={}
        self.results=[]

    def random_search(self,n_trials=10,num_iterations=5000):
        """Perform random search for hyperparameter optimization"""
        input_size=self.x_train.shape[1]
        output_size=len(np.unique(self.y_train))

        for trial in range(n_trials):
            # Random hyperparameters
            hidden_size=np.random.choice([128,256,512,1024])
            learning_rate=10**np.random.uniform(-3,-1)
            batch_size=np.random.choice([32,64,128,256])

            print(f"Trial {trial+1}/{n_trials}: hidden_size={hidden_size},lr={learning_rate:.6f},batch_size={batch_size}")

            #Create and train model
            model=NeuralNetwork(input_size,hidden_size,output_size)
            trainer=NeuralNetworkTrainer(model)

            trainer.train(self.x_train,self.y_train,self.x_test,self.y_test,num_iterations=num_iterations,learning_rate=learning_rate,eval_interval=num_iterations//2)

            #Evaluate final performance
            final_score=trainer.evaluate(self.x_test,self.y_test)

            #Store results
            trail_result={
                'hidden_size':hidden_size,
                'learning_rate':learning_rate,
                'barch_size':batch_size,
                'score':final_score,
                'model':model
            }
            self.results.append(trail_result)

            print(f"Trial {trial+1} Score:{final_score:.4f}")

            #Update best model
            if final_score>self.best_score:
                self.best_score=final_score
                self.best_model=model
                self.best_params={
                    'hidden_size':hidden_size,
                    'learning_rate':learning_rate,
                    'batch_size':batch_size
                }
                print(f"New best score: {final_score:.4f}")

        return self.best_model,self.best_params,self.results

    def save_best_model(self,filepath):
        """Save the best model found during random search"""
        if self.best_model:
            model_data=self.best_model.get_parameters()
            model_data['hyperparameters']=self.best_params
            with open(filepath,'wb') as f:
                pickle.dump(model_data,f)
            print(f'Best model saved to {filepath}')

    def load_best_model(self,filepath):
        """Load the best model from file"""
        if os.path.exists(filepath):
            with open(filepath,'rb') as f:
                model_data=pickle.load(f)

            hyperparameters=model_data.pop('hyperparameters',{})
            model=NeuralNetwork(model_data['input_size'],
                                model_data['hidden_size'],
                                model_data['output_size'])
            model.set_parameters(model_data)

            self.best_model=model
            self.best_params=hyperparameters
            self.best_score=hyperparameters.get('score',0)

            return True

        return False



