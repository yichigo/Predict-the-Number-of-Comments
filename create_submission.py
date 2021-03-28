from submission import ChallengeClassifier, generate_k_folds, accuracy
import numpy as np


def load_data():
    my_data = np.genfromtxt('train.csv', delimiter=',')
    classes = my_data[:, -1].astype(int)
    features = my_data[:, :-1]
    return features, classes

def generate_kaggle_submission():
    features, classes = load_data()
    test_data = np.genfromtxt('test.csv', delimiter=',')[:, 1:]
    myClassifier = ChallengeClassifier()
    myClassifier.fit(features, classes)
    result = myClassifier.classify(test_data)

    result_with_id = np.array([range(1, test_data.shape[0] + 1), result]).transpose()
    np.savetxt("kaggle_result.csv", result_with_id, fmt='%d', delimiter=",", header="Id,Predicted")

if __name__ == "__main__":
    generate_kaggle_submission()