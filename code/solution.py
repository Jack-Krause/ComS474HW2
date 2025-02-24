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

        show_plots(test_data, linear_svc, predictions, "linear_c", c)
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
    kernels = ['linear', 'poly', 'rbf']


    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_label = np.array(train_label)
    test_label = np.array(test_label)


    n = 1
    for kernel in kernels:
        if kernel == 'rbf':
            new_train_point = np.array([0.5, 0.5])
            new_train_label = np.array([1])
            train_data = np.vstack([train_data, new_train_point])
            train_label = np.append(train_label, new_train_label)

            test_points = np.array([[0.5, 0.5], [1.0, 0.75], [-1.0, 1.0], [0.4, 0.5]])
            test_data = np.vstack([test_data, test_points])

        kernel_svc = SVC(kernel=kernel)
        kernel_svc.fit(train_data, train_label)
        predictions = kernel_svc.predict(test_data)



        show_plots(test_data, kernel_svc, predictions, kernel, n)
        n += 1



def show_plots(test_data, svc, predictions, type_name, n):
    x_min, x_max = test_data[:, 0].min() - 1, test_data[:, 0].max() + 1
    y_min, y_max = test_data[:, 0].min() - 1, test_data[:, 0].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    scatter = plt.scatter(test_data[:, 0], test_data[:, 1], c=predictions,
                          cmap='bwr', edgecolor='k', s=50)
    plt.xlabel('Systematic Feature')
    plt.ylabel('Intensity Feature')
    plt.title(f"type: {type_name}: {n}, support vectors: {svc.n_support_}")
    plt.colorbar(scatter, ticks=[-1, 1], label='Predicted Label')
    plt.savefig(f"./plots/{type_name}{n}.jpg")
