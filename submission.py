import numpy as np
from collections import Counter
import time
# from sklearn.ensemble import RandomForestClassifier

class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = DecisionNode(None, None, lambda x : x[0] == 1)

    # TODO: finish this.
    # raise NotImplemented()
    decision_tree_root.left = DecisionNode(None, None, None, 1)
    
    A3 = DecisionNode(None, None, lambda x : x[2] == 1)
    decision_tree_root.right = A3
    
    A3_left = DecisionNode(None, None, lambda x : x[3] == 1) # check A4
    A3.left = A3_left
    A3_left.left = DecisionNode(None, None, None, 1)
    A3_left.right = DecisionNode(None, None, None, 0)

    A3_right = DecisionNode(None, None, lambda x : x[3] == 1) # check A4
    A3.right = A3_right
    A3_right.left = DecisionNode(None, None, None, 0)
    A3_right.right = DecisionNode(None, None, None, 1)

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    # TODO: finish this.
    # raise NotImplemented()
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)

    iwant = (classifier_output == 1)

    positives = true_labels[iwant]
    true_positive = positives.sum()
    false_positive = len(positives) - true_positive

    negatives = true_labels[~iwant]
    false_negative = negatives.sum()
    true_negative = len(negatives) - false_negative

    return [[true_positive, false_negative], [false_positive, true_negative]]


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """

    # TODO: finish this.
    # raise NotImplemented()
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)

    iwant = (classifier_output == 1)
    positives = true_labels[iwant]
    true_positive = positives.sum()

    return true_positive / len(positives)


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """

    # TODO: finish this.
    # raise NotImplemented()
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)

    iwant = (true_labels == 1)
    label_positives = classifier_output[iwant]
    true_positive = label_positives.sum()

    return true_positive / len(label_positives)


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """

    # TODO: finish this.
    # raise NotImplemented()
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    iwant = (classifier_output == true_labels)

    return iwant.sum() / len(true_labels)


def mean_sqrt_error(class_vector):
    """Compute the mean_sqrt_error for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the mean_sqrt_error.
    """
    # raise NotImplemented()
    if len(class_vector) == 0: # check empty class_vector
        return 0.0

    class_vector = np.array(class_vector)

    return np.sqrt( np.mean( (class_vector - np.mean(class_vector))**2 ) )


def mean_absolute_error(class_vector):
    """Compute the mean_absolute_error for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the mean_absolute_error.
    """
    # raise NotImplemented()
    if len(class_vector) == 0: # check empty class_vector
        return 0.0

    class_vector = np.array(class_vector)

    return np.mean(np.abs( class_vector - np.median(class_vector) ))


def neg_mean_error_gain(previous_classes, current_classes, error_func):
    """Compute the neg_mean_error_gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    # raise NotImplemented()
    mean_error_prev = error_func(previous_classes)

    mean_errors = []
    counts = []
    for classes in current_classes:
        mean_errors.append(error_func(classes))
        counts.append(len(classes))

    mean_error_curr = np.inner(mean_errors, counts) / np.sum(counts)

    return mean_error_prev - mean_error_curr


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit = float('inf'), min_samples_split = 2, min_samples_leaf = 1, criterion = 'mse'):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        if criterion == 'mae':
            self.error_func = mean_absolute_error
        elif criterion == 'mse':
            self.error_func = mean_sqrt_error


    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        self.root = self.__build_tree__(features, classes)


    def find_split_threshold(self, feature, classes): # for single sorted feature
        feature_unique = np.unique(feature) # unique() automatically sort the result
        if len(feature_unique) == 1: 
            return 0.0, feature_unique[0]

        thresholds = (feature_unique[1:] + feature_unique[:-1]) / 2.0
        neg_mean_error_gain_max = 0.0
        threshold_best = thresholds[0]
        for threshold in thresholds:
            i_true = (feature >= threshold)

            # check min_samples_leaf
            num_true = np.sum(i_true)
            if (num_true < self.min_samples_leaf) | (len(classes) - num_true < self.min_samples_leaf):
                continue

            neg_mean_error_gain_curr = neg_mean_error_gain(classes, [classes[i_true], classes[~i_true]], error_func=self.error_func)
            if neg_mean_error_gain_curr > neg_mean_error_gain_max:
                neg_mean_error_gain_max = neg_mean_error_gain_curr
                threshold_best = threshold

        return neg_mean_error_gain_max, threshold_best


    def __build_tree__(self, features, classes, depth = 0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.
        # raise NotImplemented()
        if np.all(classes == classes[0]):
            return DecisionNode(None, None, None, classes[0])

        n_samples = len(classes)
        if (depth >= self.depth_limit) | (n_samples < self.min_samples_split) | (n_samples < 2*self.min_samples_leaf):
            return DecisionNode(None, None, None, np.median(classes))

        # find best split feature and threshold
        i_best = 0
        neg_mean_error_gain_max = 0.0
        threshold_best = 0.0
        idx = np.argsort(features, axis = 0) # index of each sorted columns
        for i in range(features.shape[1]):
            neg_mean_error_gain_curr, threshold = self.find_split_threshold(features[idx[:,i],i], classes[idx[:,i]])
            
            if neg_mean_error_gain_curr > neg_mean_error_gain_max:
                i_best = i
                neg_mean_error_gain_max = neg_mean_error_gain_curr
                threshold_best = threshold

        # if not split, return the average
        if neg_mean_error_gain_max == 0.0:
            return DecisionNode(None, None, None, np.median(classes))
        else: # do split
            i_true =  (features[:,i_best] >= threshold_best)
            node = DecisionNode(None, None, lambda x : x[i_best] >= threshold_best)
            node.left  = self.__build_tree__(features[ i_true], classes[ i_true], depth=depth+1)
            node.right = self.__build_tree__(features[~i_true], classes[~i_true], depth=depth+1)

        return node

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """

        class_labels = [self.root.decide(feature) for feature in features]

        # TODO: finish this.
        # raise NotImplemented()\
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """

    # TODO: finish this.
    # raise NotImplemented()
    X, y = dataset
    N = len(y)

    # generate random indices
    idx = np.array(range(N))
    np.random.shuffle(idx)

    # shuffle dataset
    X = X[idx]
    y = y[idx]

    # find split points of indices
    splits = [(N//k)*i for i in range(k+1)] # drop the tail

    # generate k fold
    k_folds = []
    for i in range(k):
        X_train = np.append(X[:splits[i]], X[splits[i+1]:], axis = 0)
        y_train = np.append(y[:splits[i]], y[splits[i+1]:], axis = 0)
        X_test = X[ splits[i]:splits[i+1] ]
        y_test = y[ splits[i]:splits[i+1] ]
        k_folds.append(( (X_train, y_train), (X_test, y_test) ))

    return k_folds


class RandomForest: # modify to regressor
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit,
                 example_subsample_rate,
                 attr_subsample_rate,
                 min_samples_split = float('inf'),
                 min_samples_leaf = 1,
                 criterion = 'mse'):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.attr_dict = {}

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        # raise NotImplemented()
        N, N_feature = np.shape(features)
        if self.example_subsample_rate <= 1:
            N_example = round(N * self.example_subsample_rate)
        else:
            N_example = self.example_subsample_rate

        if self.attr_subsample_rate <= 1:
            N_attr = round(N_feature * self.attr_subsample_rate)
        else:
            N_attr = self.attr_subsample_rate

        for i in range(self.num_trees):
            tree = DecisionTree(depth_limit = self.depth_limit,
                                min_samples_split = self.min_samples_split,
                                min_samples_leaf = self.min_samples_leaf,
                                criterion = self.criterion)
            # example subsample
            i_example = np.random.choice(N, N_example, replace = False) #True) # I think False is better
            # attribute subsample
            i_attr = np.random.choice(N_feature, N_attr, replace = False)
            self.attr_dict[i] = i_attr
            # fit each tree
            # slice a matrix, must separate the process of row and column
            tree.fit(features[i_example,:][:,i_attr], classes[i_example])
            self.trees.append(tree)


    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """

        # TODO: finish this.
        # raise NotImplemented()
        classes = []
        for i, tree in enumerate(self.trees):
            i_attr = self.attr_dict[i]
            classes.append(tree.classify(features[:,i_attr]))

        if self.criterion == 'mse':
            classes = np.mean(classes, axis = 0) # mean is better for mse
        elif self.criterion == 'mae':
            classes = np.median(classes, axis = 0) # median is better for mae

        return classes



class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, num_trees = 47,
                        depth_limit = 22,
                        min_samples_split = 10,
                        min_samples_leaf = 5,
                        example_subsample_rate = 0.4,
                        attr_subsample_rate = 0.8,
                        criterion = 'mae',
                        # only fit the first 26 features here
                        n_features = 26,
                        random_seed = 9
                        ):
        """Create challenge classifier.
        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        # raise NotImplemented()\
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.criterion = criterion
        self.n_features = n_features
        self.random_seed = random_seed
        np.random.seed(seed = random_seed)

        self.trees = RandomForest(num_trees = self.num_trees,
                                  depth_limit = self.depth_limit,
                                  min_samples_split = self.min_samples_split,
                                  min_samples_leaf = self.min_samples_leaf,
                                  example_subsample_rate = self.example_subsample_rate,
                                  attr_subsample_rate = self.attr_subsample_rate,
                                  criterion = self.criterion
                                  )

    def fit(self, features, classes):
        """Build the underlying tree(s).
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        # raise NotImplemented()
        self.trees.fit(features[:,:self.n_features], classes)


    def classify(self, features):
        """Classify a list of features.
        Classify each feature in features as either 0 or 1.
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """

        # TODO: finish this.
        # raise NotImplemented()
        
        return self.trees.classify(features[:,:self.n_features])


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        # TODO: finish this.
        # raise NotImplemented()
        return data*data + data


    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row
        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.
        # raise NotImplemented()
        row_sum = data[:100].sum(axis = 1)
        max_sum_index = np.argmax(row_sum)
        return row_sum[max_sum_index], max_sum_index


    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        # TODO: finish this.
        # raise NotImplemented()
        unique_keys, unique_counts = np.unique(data[data>0], return_counts = True)
        return list(zip(unique_keys, unique_counts))



def return_your_name():
    # return your name
    # TODO: finish this
    # raise NotImplemented()
    return "Yichao Zhang"
