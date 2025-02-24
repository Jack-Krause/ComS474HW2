import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def svm_with_diff_c(train_label, train_data, test_label, test_data):
    '''
    Use different value of cost c to train a svm model. Then apply the trained model
    on testing label and data.
    
    The value of cost c you need to try is listing as follow:
    c = [0.01, 0.1, 1, 2, 3, 5]
    Please set kernel to 'linear' and keep other parameter options as default.
    No return value is needed
    '''

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    ### YOUR CODE HERE
    costs = [0.01, 0.1, 1, 2, 3, 5]
    for c in costs:
        linear_svc = SVC(kernel='linear', C=c)
        linear_svc.fit(train_data, train_label)
        predictions = linear_svc.predict(test_data)

        x_min, x_max = test_data[:, 0].min() - 1, test_data[:, 0].max() + 1
        y_min, y_max = test_data[:, 0].min() - 1, test_data[:, 0].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 500),
            np.linspace(y_min, y_max, 500)
        )

        Z = linear_svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        scatter = plt.scatter(test_data[:, 0], test_data[:, 1], c=predictions,
                              cmap='bwr', edgecolor='k', s=50)
        plt.xlabel('Systematic Feature')
        plt.ylabel('Intensity Feature')
        plt.title(f"Test data: {c}, support vectors: {linear_svc.n_support_}")
        plt.colorbar(scatter, ticks=[-1, 1], label='Predicted Label')
        plt.savefig(f"./plots/plot{c}.jpg")



    ### END YOUR CODE
    

def svm_with_diff_kernel(train_label, train_data, test_label, test_data):
    '''
    Use different kernel to train a svm model. Then apply the trained model
    on testing label and data.
    
    The kernel you need to try is listing as follow:
    'linear': linear kernel
    'poly': polynomial kernel
    'rbf': radial basis function kernel
    Please keep other parameter options as default.
    No return value is needed
    '''

    ### YOUR CODE HERE



    ### END YOUR CODE
