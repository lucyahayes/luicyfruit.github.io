---
layout: post
title: "Predicting Cover-Type of a Forest Using Random Forests
date: 2019-11-11
---

# Predicting Cover-Type of a Forest

This project explores using Machine Learning methods to predict the type of forest cover an area would have, given parameters such as soil type, elevation, distance to water, and amount of shade. The dataset was obtained from the UCI Machine Learning Database. I was particularly drawn to this dataset because it is an example of using Machine Learning in the ecology world, where I understand the amount of field work that goes into obtaining these samples and datapoints. Aka, wouldn't it be great to use ML to scale down on some of that manual work!

---
## Exploring the Dataset

At initial glance, the dataset contained no column names. This required doing some more research to find what each column corresponded to, as it was not contained in the info file of the dataset. After finding the labels for the 55(!) columns, I looked into each of the columns. Although they were all the same datatype (non-null int64), the columns that were already one-hot encoded were binary. I decided to use the `pandas profiling` package to explore the continuous variables. 

The first observations are that there are a few highly corelated variables, along with some of the variables producing significant skew.

![Skew](https://luicyfruit.github.io/img/skew.svg) 
![Correlation](https://luicyfruit.github.io/img/correlation.svg)

Since we will be working with Random Forests and Decision Trees, we can leave this for now. However, it is important to note that we would have to check the assumptions of other algorithms for these cases if we were to use this dataset in the future.

Next I decided, that for computational time, I would be using a sample of the initial dataset, as the initial dataset was 500,000+ rows. I took a random sample of 50,000 rows
```py
sample_df = df.sample(n = 50000)
```

My final cause for concern was the frequency of the various different cover types. Looking at the initial distribution, there was a very low occurance of some of the cover types
![Cover_Type](https://luicyfruit.github.io/img/distro.svg)

When taking the sample, the occurrence was even lower:

| Cover Type | Occurrence |
| ------ | ----------- |
| 1 | 18077 |
| 2 | 24518 |
| 3 | 3125 |
| 4 | 244 |
| 5 | 836 |
| 6 | 1466 |
| 7 | 1734 |

I decided, that for this run, I would keep all the Cover types, even given the low occurence of Type 4. However, if the model was performing too poorly, this would be something to look into. 

Lastly, I dropped the target variable (CoverType), and split the dataset into a training and testing set. 

```py
target = sample_df['Cover_Type']
sample_df.drop("Cover_Type", axis=1, inplace=True)
data_train, data_test, target_train, target_test = train_test_split(sample_df, target, 
                                                                   test_size = 0.25, 
                                                                   random_state=119)                       
```
---

## Building a Regular Decision Tree for Baseline 

I decided to build a regular decision tree for a baseline reference for my later Random Forest Model. I started by initializing a DecisionTree Object with no parameters. I then looked at the accuracy score for the fit of the tree, which came out to 81.4% (not bad for a first run!). 

``` py
tree_clf = DecisionTreeClassifier() 
tree_clf.fit(data_train, target_train)
pred = tree_clf.predict(data_test)
accuracy_score(target_test, pred)
```
`0.8144`


Next I used `GridSeachCV` to hypertune the parameters of the model, and also as basic practice for my later Random Forest Model. I used the following parameter grid:
``` py
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6]
    }
```
Gridsearch Found the best Parameters to be:
'criterion': 'entropy',
 'max_depth': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2
After feeding a tree those parameters and looking at the accuracy score again, the testing accuracy only went up to 82%. Not much improvement, but good practice.The final feature importances were 
![FeatureImportanceTree](https://luicyfruit.github.io/img/feature_importance_tree.png)

This shows the most important feature to be elevation, with the continious variables having more importnace than the binary variables (wilderness type and soil type). 

---

## Building a Random Forest Model

Now that the initial Decision Tree was built as a baseline for success, I wanted to use a Random Forest Model to see if that improved predictions. Once again, to start I ran a Random Forest Model with no parameters, and looked at the testing and training scores. 
``` py
forest = RandomForestClassifier()
forest.fit(data_train, target_train)
forest.score(data_test, target_test)
```
`.0846`
Our inital testing accuracy score is 84.6%. Looking at our feature importances:

![FeatureImportanceForest1](https://luicyfruit.github.io/img/feature_importance_forest1.png)

We can see that although elevation still remains the most important variable, the importance of some of the soil types and wilderness areas have increased from the baseline decision tree. 

---
Next I used GridSearch to finetune the parameters of the model. As before, I passed a testing parameter grid, and since this time it much harder computationally, I also wanted to see how long the search would take, as I am only using a sample of the original dataset. 
``` py
rf_param_grid = {
    'n_estimators': [10, 30, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 6, 10],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [1, 2, 5]
}
start = time.time()
rf_grid_search = GridSearchCV(forest_2, rf_param_grid, cv=3)
rf_grid_search.fit(data_train, target_train)
```
`Testing Accuracy: 84.32%`
`Total Runtime for Grid Search on Random Forest Classifier: 410.1 seconds`
I then passed a the optimal parameters given by Gridsearch to a new RandomForestClassifier
```py
forest_cv = RandomForestClassifier(criterion = 'entropy', max_depth = None, min_samples_leaf = 1,
                                min_samples_split = 10, n_estimators=100)
forest_cv.fit(data_train, target_train)
```
This left me with a score of 86.4%, which was 1.8% better than the original random forest, and 5% better than the origial decision tree. The feature importances also showed some changes:
![FeatureImportanceForest1](https://luicyfruit.github.io/img/feature_importance_forest1.png)

Elevation consistently remained the most important, however the relative importances of the continuous variables increased from the baseline RandomForest. Additionally, the importances of the soil types decreased.

## Conclusion
While this only used a sample of the original dataset, we were able to achieve 86% accuracy in the predictions! Future modeling would include other algorithms and corresponding data cleaning, and also use of a larger dataset. 









