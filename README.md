1. Load the dataset: Loaded the cleaned dataset and defined features and labels
   Feature matrix X: Dropped the Regression target column (final_score) and Classification target column (passed)
   Regression label (y_reg): final_score column is the Regression label
   Classification label (y_clf): passed column is the Classification label
2. Encode categorical columns: For each categorical column in X, we performed Label encoding as the categories have a natural order
   gender = Male: 0, Female: 1
   internet_access = Yes: 1, No: 0
   extracurricular = Yes: 1, No: 0
   If the categories have no natural order then one-hot encoding will be applied. One-hot encoding will represent each category as separate binary feature and hence it avoids the false-              ordinal-relationship. Label encoding will assign arbitrary integers to features. Here the categories have natural orders and hence we implemented Label encoding
3. Leak-free train-test split and scaling: Splitted X and both labels into training and test sets. Fitted a StandardScaler only on the training features. Transformed both X_train and X_test using the fitted scaler. If we fit the scaler on the entire dataset before splitting it will cause data leakage because the computed mean and standard deviation will include information from the test set. If the data leakage happens, the model will perform better during training and test phase but not in live. So if we fit the scaler only on the training set, the test set remains completely unseen during training and provides better result
4. Regression model — Linear Regression: 
Trained a linear regression model
Computed MSE and R² for linear regression model
Printed the model's coefficients alongside the corresponding feature names. Identified the three features with the largest absolute coefficient values (study_hours_per_week, attendance_rate, previous_score).
A large positive coefficient means when the scaled feature increases by one unit that is one standard deviation above the mean after scaling, the predicted target value increases by the coefficient amount. In this scenario if a student’s study hours per week increase by one unit, the model predicts their final score will rise by about 12.8 points
A large negative coefficient means when the scaled feature decreases by one unit that is one standard deviation below the mean after scaling, the predicted target value decreases by the coefficient amount.
Trained a ridge regression model. Computed MSE and R² for ridge regression model and compared it against linear regression model
Ridge regression will give different coefficient values compared to OLS linear regression because it will add penalty and try to shrink the coefficient values smaller when the features are highly correlated to avoid overfitting. The alpha parameter controls the strength of penalty applied that is when alpha = 0, ridge behaves like OLS linear regression and when larger values of alpha applied it will make the coefficient smaller and improve the performance if the model is overfitting
In this scenario Ridge Regression performs almost identical to Linear Regression as there is no high multicollinearity
5. Classification model — Logistic Regression: Checked y_clf_train.value_counts and both the classes have above 35% of samples and hence did not perform imbalance handling.
Trained a Logistic Regression model and computed confusion matrix, accuracy, precision, recall, and F1 score
Computed ROC Curve. Calculated AUC score. Plotted the ROC curve

a) Formula:
   Precision = TP/(TP+FP), TP = True Positives, FP = False Positives. It measures how many of the predicted positives are actually correct
   Recall = TP/(TP+FN), FN = False Negatives. It measures how many of the actual positives were correctly identified
b) In this scenario, Recall metric is important. False Negatives are costlier than False Positives because if a passed student is marked as failed then it will cause serious problem. If a failed student is marked as passed (False Positives) it is not fair but this will be less harmful compared to False Negatives
c) The AUC value measures the model’s ability to separate the two classes. AUC Score of 0.5 means no better than random guessing and an AUC Score of 1.0 means perfect separation. Here we have an AUC Score of 0.93 and it has great ability to separate the two classes.

b. Decision-threshold sensitivity: For the logistic regression model trained, we varied the decision threshold from 0.30 to 0.70 in steps of 0.10. At each threshold, we computed precision, recall, and F1-score. Printed all five rows as a table with columns: Threshold | Precision | Recall | F1
a) Formula:
Precision = TP/(TP+FP), TP = True Positives, FP = False Positives. It measures how many of the predicted positives are actually correct
Recall = TP/(TP+FN), FN = False Negatives. It measures how many of the actual positives were correctly identified
b) At threshold of 0.40 we have maximum F1-score of 0.900
c) In this scenario, Recall metric is important. False Negatives are costlier than False Positives because if a passed student is marked as failed then it will cause serious problem. If a failed student is marked as passed (False Positives) it is not fair but this will be less harmful compared to False Negatives
d) If we lower the threshold, Recall will get better but there will be more False Positives. So it is better to calculate Recall with average threshold

6. Regularization experiment on Logistic Regression: Trained a second logistic regression with C=0.01. Compared its precision, recall, and AUC against the baseline model with C=1.0.
In logistic regression, the parameter C controls the strength of regularization which is the inverse of the penalty term applied to the coefficients. A larger C = 1.0 means weaker regularization allowing the model more flexibility to fit the training data while a smaller C = 0.01 enforces stronger L2 penalty, shrinking coefficients and simplifying the model. Here, reducing C from 1.0 to 0.01 led to higher precision but lower recall and AUC remained nearly identical. This indicates that with strong regularization it was more accurate when it predicted a pass but it missed many true passes. Since recall is more important here, with C = 0.01 overall performance is low though the precision is increased.

7. Bootstrap confidence interval for AUC difference: Calculated AUC score of base model and strong model for Bootstrap and computed the AUC difference between them. Computed the mean AUC difference, 2.5th and 97.5th percentile of the 500 difference values.
Mean AUC difference:  6.2499999999998e-05
2.5th percentile:  -0.01651475694444441
97.5th percentile:  0.01672092013888891
95% Confidence Interval: -0.01651475694444441, 0.01672092013888891
95% confidence interval for the AUC difference includes zero and it means base model (C=1.0) does not show a consistently reliable advantage over the strong model (C=0.01) and so the difference is not reliable
