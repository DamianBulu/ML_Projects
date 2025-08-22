
import matplotlib.pyplot as plt
import numpy as np

class TrainingVisualizer():
    def __init__(self):
        self.fig,self.axes=None,None
    def plot_training_history(self,training_history,save_path=None):
        """Plot training loss and accuracy history"""
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))

        #Plot loss
        iterations,losses=zip(*training_history['loss'])
        ax1.plot(iterations,losses,'b-',label='Loss')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)

        #Plot accuracy
        iterations_acc,train_acc=zip(*training_history['accuracy'])
        _,test_acc=zip(*training_history['test_accuracy'])
        ax2.plot(iterations_acc,train_acc,'g-',label='Train Accuracy')
        ax2.plot(iterations_acc,test_acc,'r-',label='Test Accuracy')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Test Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path,dpi=300,bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self,model,x_test,y_test,save_path=None):
        """Plot confusion matrix for model predictions"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        predictions=[]
        for i in range(len(x_test)):
            predictions.append(model.predict(x_test[i]))

        cm=confusion_matrix(y_test,predictions)

        plt.figure(figsize=(10,8))
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')

        if save_path:
            plt.savefig(save_path,dpi=300,bbox_inches='tight')

        plt.show()
