
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

