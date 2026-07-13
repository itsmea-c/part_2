import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\saran\Documents\AIML\part_1\cleaned_data.csv")   # load the cleaned dataset
X = df.drop(columns=["final_score", "passed"]) # feature matrix
y_reg = df["final_score"]      # regression label         
y_clf = df["passed"]  # classification label
X["gender"] = X["gender"].map({"Male": 0, "Female": 1})  # encoding
X["internet_access"] = X["internet_access"].map({"Yes": 1, "No": 0})
X["extracurricular"] = X["extracurricular"].map({"Yes": 1, "No": 0})
y_clf = y_clf.map({"Yes": 1, "No": 0})
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42) # splitting the data
scaler = StandardScaler()   # scaling
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
linear_regression_model = LinearRegression()   # training a linear regression model
linear_regression_model.fit(X_train_scaled, y_reg_train)
y_pred_reg_linear = linear_regression_model.predict(X_test_scaled)
mse_linear = mean_squared_error(y_reg_test, y_pred_reg_linear)
r2_linear = r2_score(y_reg_test, y_pred_reg_linear)
print("Linear Mean Squared Error: ", mse_linear) # mse linear regression
print("Linear R2: ", r2_linear)  # r2 linear regression
coefficient_df = pd.DataFrame({"Feature": X.columns, "Coefficient": linear_regression_model.coef_ , "Absolute Coefficient": np.abs(linear_regression_model.coef_)})
print("Model Coefficients: ", coefficient_df)  # Model coefficients
top3_features = coefficient_df.sort_values(by="Absolute Coefficient", ascending=False).head(3)
print("Three features with the largest absolute coefficient values: ", top3_features)  # three features with the largest absolute coefficient values
ridge_model = Ridge(alpha=1.0)   # training a ridge regression model
ridge_model.fit(X_train_scaled, y_reg_train)
y_pred_reg_ridge = ridge_model.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_reg_test, y_pred_reg_ridge)  
r2_ridge = r2_score(y_reg_test, y_pred_reg_ridge)   
print("Ridge Mean Squared Error: ", mse_ridge) # mse ridge regression
print("Ridge R2: ", r2_ridge)  # r2 ridge regression
comparison = pd.DataFrame({"Model": ["Linear Regression", "Ridge Regression (α=1.0)"], "MSE": [mse_linear, mse_ridge], "R²": [r2_linear, r2_ridge]})
print("Comparison Table: ", comparison)  # comparison of linear regression and ridge regression
y_clf_train_counts = y_clf_train.value_counts()  # count of y classification train values
print("Count of y_clf_train: ", y_clf_train_counts)
logistic_regression_model = LogisticRegression(max_iter=1000, random_state=42)   # training a logistic regression model
logistic_regression_model.fit(X_train_scaled, y_clf_train)
y_clf_pred = logistic_regression_model.predict(X_test_scaled)
accuracy = accuracy_score(y_clf_test, y_clf_pred)
conf_matrix = confusion_matrix(y_clf_test, y_clf_pred)
class_report = classification_report(y_clf_test, y_clf_pred)
print("Accuracy: ", accuracy)
print("Confusion Matrix: ", conf_matrix)
print("Classification Report: ", class_report)
y_clf_prob = logistic_regression_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_clf_test, y_clf_prob)   # roc curve
auc_score = roc_auc_score(y_clf_test, y_clf_prob)
print("AUC Score: ", auc_score)
plt.figure(figsize=(6,6))   # plotting roc curve
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})", color="green")
plt.plot([0,1], [0,1], linestyle="--", color="blue")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.show()
thresholds = np.arange(0.30, 0.80, 0.10)  # varying the threshold
print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
print("-"*50)
for threshold in thresholds:
    y_clf_pred_threshold = (y_clf_prob >= threshold).astype(int)
    precision = precision_score(y_clf_test, y_clf_pred_threshold)
    recall = recall_score(y_clf_test, y_clf_pred_threshold)
    f1 = f1_score(y_clf_test, y_clf_pred_threshold)
    print(f"{threshold:<10.2f} | {precision:<10.3f} | {recall:<10.3f} | {f1:<10.3f}")
logistic_regression_model_strong = LogisticRegression(C=0.01, max_iter=1000, random_state=42)   # training a second logistic regression model
logistic_regression_model_strong.fit(X_train_scaled, y_clf_train)
y_clf_pred_strong = logistic_regression_model_strong.predict(X_test_scaled)
accuracy_strong = accuracy_score(y_clf_test, y_clf_pred_strong)
conf_matrix_strong = confusion_matrix(y_clf_test, y_clf_pred_strong)
class_report_strong = classification_report(y_clf_test, y_clf_pred_strong)
print("Accuracy Strong: ", accuracy_strong)
print("Confusion Matrix Strong: ", conf_matrix_strong)
print("Classification Report Strong: ", class_report_strong)
y_clf_prob_strong = logistic_regression_model_strong.predict_proba(X_test_scaled)[:, 1]
fpr_strong, tpr_strong, thresholds = roc_curve(y_clf_test, y_clf_prob_strong)   # roc curve for strong model
auc_score_strong = roc_auc_score(y_clf_test, y_clf_prob_strong)
print("AUC Score Strong: ", auc_score_strong)
plt.figure(figsize=(6,6))   # plotting roc curve for strong model
plt.plot(fpr_strong, tpr_strong, label=f"ROC Curve for strong model (AUC = {auc_score_strong:.3f})", color="green")
plt.plot([0,1], [0,1], linestyle="--", color="blue")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Strong Logistic Regression")
plt.legend(loc="lower right")
plt.show()
thresholds = np.arange(0.30, 0.80, 0.10)  # varying the threshold for strong model
print(f"{'Threshold':<10} | {'Precision Strong':<10} | {'Recall Strong':<10} | {'F1 Strong':<10}")
print("-"*50)
for threshold in thresholds:
    y_clf_pred_threshold_strong = (y_clf_prob_strong >= threshold).astype(int)
    precision_strong = precision_score(y_clf_test, y_clf_pred_threshold_strong)
    recall_strong = recall_score(y_clf_test, y_clf_pred_threshold_strong)
    f1_strong = f1_score(y_clf_test, y_clf_pred_threshold_strong)
    print(f"{threshold:<10.2f} | {precision_strong:<10.3f} | {recall_strong:<10.3f} | {f1_strong:<10.3f}")  # precision, recall and F1 score of strong model
print(f"{'Model':<20} | {'Precision':<10} | {'Recall':<10} | {'AUC':<10}")   # Comparing precision, recall and AUC of baseline model and strong model
print("-"*60)
print(f"{'C=1.0 (baseline model)':<20} | {precision:<10.3f} | {recall:<10.3f} | {auc_score:<10.3f}")
print(f"{'C=0.01 (strong model)':<20} | {precision_strong:<10.3f} | {recall_strong:<10.3f} | {auc_score_strong:<10.3f}")
n_bootstrap = 500  # bootstrap
auc_differences = []
for i in range(n_bootstrap):
    sample_indices = np.random.choice(len(y_clf_test), size=len(y_clf_test), replace=True)
    y_clf_test_sample = y_clf_test.iloc[sample_indices]
    y_clf_prob_sample = y_clf_prob[sample_indices]
    y_clf_prob_strong_sample = y_clf_prob_strong[sample_indices]    
    auc_base = roc_auc_score(y_clf_test, y_clf_prob_sample)  # auc score of base model for bootstrap
    auc_strong = roc_auc_score(y_clf_test, y_clf_prob_strong_sample)  # auc score of strong model for bootstrap
    auc_differences.append(auc_base - auc_strong)  # auc difference between base and strong model for bootstrap
auc_differences = np.array(auc_differences)
mean_auc_difference = np.mean(auc_differences)   # mean auc difference
ci_lower = np.percentile(auc_differences, 2.5)   # 2.5th percentile
ci_upper = np.percentile(auc_differences, 97.5)  # 97.5th percentile
print("Mean AUC difference: ", mean_auc_difference)
print("2.5th percentile: ", ci_lower)
print("97.5th percentile: ", ci_upper)
print("95% CI:", (ci_lower, ci_upper))    # 95% confidence interval