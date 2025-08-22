

import numpy as np
from data_processor import DataProcessor
from model import NeuralNetwork
from trainer import NeuralNetworkTrainer
from visualizer import TrainingVisualizer
from hyperparameter_tuner import HyperparameterTuner

def main():

    #Load data
    print("Loading MNIST data...")
    data_processor=DataProcessor()
    x_train,x_test,y_train,y_test=data_processor.load_mnist_data()

    stats=data_processor.get_data_stats()
    print(f"Data loaded:{stats['train_samples']} train,{stats['test_samples']} test samples")
    print(f"Input dimension: {stats['input_dim']}, Classes: {stats['num_classes']}")

    #Hyperparameter tuning with random search
    print("\nStarting hyperparameter tuning with random search...")
    tuner=HyperparameterTuner(x_train,y_train,x_test,y_test)
    best_model,best_params,all_results=tuner.random_search(n_trials=10,num_iterations=10000)

    print(f"\nBest parameters: {best_params}")
    print(f"\nBest accuracy: {tuner.best_score:.4f} ")

    #Save best model
    tuner.save_best_model('best_model.pkl')

    #Train final model with best parameters
    print("\nTraining final model with best parameters...")
    final_model=NeuralNetwork(stats['input_dim'],best_params['hidden_size'],stats['num_classes'])

    trainer=NeuralNetworkTrainer(final_model)
    trainer.train(x_train,y_train,x_test,y_test,num_iterations=20000,learning_rate=best_params['learning_rate'])

    #Evaluate final model
    final_acc=trainer.evaluate(x_test,y_test)
    print(f"Final test accuracy: {final_acc:.4f}")

    #Visualize results
    visualizer=TrainingVisualizer()
    visualizer.plot_training_history(trainer.training_history,'training_history.png')
    visualizer.plot_confusion_matrix(final_model,x_test[:1000],y_test[:1000],'confusion_matrix.png')

if __name__=="__main__":
    main()
