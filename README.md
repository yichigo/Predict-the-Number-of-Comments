
# Predict the Number of Comments an Article will Receive by Using Random Forest


## Data Description

In this challenge, we want to predict the number of comments an article on the Web will receive in its comment section within the next 24 hours.

We have crawled data from many websites and calculated number of comments posted before and after various base timestamps. Other features include the number of comments the article has received in the 24 and 48 hours prior to the base timestamp, average number of comments per article for the website where the article is posted on, etc.

The files we provide you include

    train.csv: the training set
    test.csv: the unlabeled test set
    create_submission.py: the script to create Kaggle submission file

Each column in train.csv are as follows:

    1 to 20: Mean, standard deviation, min, max and median of the attributes 21 to 24 for the website that the article is posted on.

    21: Total number of comments the article has received so far.

    22: Number of comments in the past day prior to the base timestamp of this sample.

    23: Number of comments in the day before yesterday, prior to the base timestamp of this sample.

    24: Number of comments in the first day after the article is posted.

    25: The length of time between when the article is posted and the base timestamp.

    26: The length of the article.

    27 to 226: The 200 bag of words features for a list of 200 frequent words (word labels are not provided).

    227 to 233: whether the base timestamp is a Monday, Tuesday, ... Sunday. 0 = No, 1 = Yes.

    234 to 240: whether the article was posted on a Monday, Tuesday, ... Sunday. 0 = No, 1 = Yes.

    241: The label: number of comments in the next 24 hours from the base timestamp.
    test.csv has an additional ID column at the beginning of each row.

You are free to change create_submission.py to incorporate k-fold cross validation or anything else you want.


## Evaluation

We will use Mean Absolute Error for evaluation.


## Model
Random Forest Regressor

    submission.py

This code is a self-written Random Forest Classifier. Since the competetion uses mean absolute error to evaluate the performance, I did the following modification:

1) Criterion: mean absolute error

2) Leaf Value: we return the median value as output,  because the median value minimize the mean absolute error

3) Absolute Error: should be |value - median value|, not |value - mean value|

4) Node Splitting: we tested the all the split thresholds for each feature, and chose the best feature and the best threshold which maximize the negative mean absolute error gain

5) Example Subsample: we set the replace = False, which is different with the traditional random forest, because we have enough data here

6) Ensemble: we use median ensemble rather than average ensemble, which was used in traditional random forest. Because the criterion is mean absolute error here. We found that median ensemble is better than average ensemble in almost every time.


Feature Selection:

We only fit the first 26 features here, since all the other features seems useless


Other Data Preprocessing / Transformations:

None


Performance vs Scikit-Learn:

1) Decision Tree: our single decision tree model returns exactly the same result with the DecisionTreeRegressor of scikit-learn

2) Random Forest: our random forest model beats the RandomForestRegressor of scikit-learn in almost every time, because scikit-learn do not provide a median ensemble choice. But if we redo the median ensemble for RandomForestRegressor in scikit-learn, then they have similar performances


Running Time vs Scikit-Learn:

Our random forest model is much slower than Scikit-Learn, since we do not use Cython here. With the same hyper parameters, scikit learn spends about tens of minutes to finish the training, but our model needs about hours.


Speed Up the Random Forest:

There is a very efficient way to speed up the random forest. The running time is mainly due to the sorting and checking all the threshold of continuous attributes. In a traditional regression decision tree node, to find the best splitting threshold for each continuous feature, we have to calculate the unique sorted feature values $[a_0, a_1 ...]$, and the thresholds are chosen from  $[(a_0 + a_1)/2, (a_1 + a_2)/2, ... ]$, the time complexity is $O(N log N)$ for sorting and $O(N^2)$ to calculate the MAEs of all the thresholds.     But if the "best" is not so necessary, why don't we simply split the range between $[a_min, a_max]$ by maybe 10 thresholds, just like LightGBM did. Then the time complexity is reduced to $O(N)$ to generate the thresholds and $O(10 N)$ to calculate the MAEs.  So the RF flies, and you need about only minutes for every training. Then it becomes available to do the cross validation and fine tune the hyper parameter.

The speed up version of random forest model can are written in:

    submission_light.py



Default Hyper Parameters:
    num_trees = 47,
    depth_limit = 22,
    min_samples_split = 10,
    min_samples_leaf = 5,
    example_subsample_rate = 0.4,
    attr_subsample_rate = 0.8,
    criterion = 'mae',
    random_seed = 9




    
