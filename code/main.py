from helper import load_features
from solution import svm_with_diff_c, svm_with_diff_kernel
import os


def test_svm():
    traindataloc, testdataloc = "./data/train.txt", "./data/test.txt"

    if os.path.isfile(traindataloc) and os.path.isfile(testdataloc):
        train_data, train_label = load_features(traindataloc)
        test_data, test_label = load_features(testdataloc)

        svm_with_diff_c(train_label.tolist(), train_data.tolist(),
                        test_label.tolist(), test_data.tolist())
        svm_with_diff_kernel(train_label.tolist(), train_data.tolist(),
                             test_label.tolist(), test_data.tolist())

    else:
        print("FILE NOT FOUND")


if __name__ == '__main__':
    test_svm()
