
from import_package import *
# Split the dataset into training and testing sets
X_train_XGBC, X_test_XGBC, y_train_XGBC, y_test_XGBC = train_test_split(X_new_train ,y, test_size=0.3, random_state=42)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote_xg = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote_xg.fit_resample(X_train_XGBC, y_train_XGBC)

# Initialize the XGBoost classifier with specified hyperparameters
XGBC = XGBClassifier(
    n_estimators=12,  # Number of boosting stages (trees)
    max_depth=5,  # Maximum depth of the trees
    learning_rate=0.15,  # Step size at each boosting iteration
    reg_lambda=10,  # L2 regularization
    reg_alpha=2,  # L1 regularization
    objective='binary:logistic'  # Logistic regression for binary classification
)

# Train the XGBoost model on the balanced training data
XGBC.fit(X_train_bal, y_train_bal)

# Make predictions on the test set using the trained XGBoost model
y_predict_xg = XGBC.predict(X_test_XGBC)
y_predict_proba_xg = XGBC.predict_proba(X_test_XGBC)[:, 1]  # Probabilities for the positive class
y_train_predict_proba_xg = XGBC.predict_proba(X_train_bal)[:, 1]
roc_auc_score_train_xg = roc_auc_score(y_train_bal, y_train_predict_proba_xg)
print("ROC AUC SCORE TRAIN", roc_auc_score_train_xg)

# Evaluate the model using accuracy, confusion matrix, and ROC AUC score
conf_matrix_xg = confusion_matrix(y_test_XGBC, y_predict_xg)
accuracy_xg = accuracy_score(y_test_XGBC, y_predict_xg)
roc_auc_score_xg = roc_auc_score(y_test_XGBC, y_predict_proba_xg)
classification_report_xg = classification_report(y_test_XGBC, y_predict_xg)

# Print the evaluation results
print("accuracy_xg = ", accuracy_xg)
print("conf_matrix_xg = ", conf_matrix_xg)

# Display the confusion matrix as a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_xg, display_labels=["Negative Class", "Positive Class"])
disp.plot(cmap="Blues")

# Print the ROC AUC score for test set
print("roc_auc_score_xg = ", roc_auc_score_xg)

# Print the classification report for the test set (precision, recall, f1-score)
print("Classification Report (Test):")
print(classification_report_xg)

# Classification report for training data
y_train_predict_xg = XGBC.predict(X_train_bal)
classification_report_train_xg = classification_report(y_train_bal, y_train_predict_xg)
print("Classification Report (Train):")
print(classification_report_train_xg)


# ROC curve for training data
fpr_train, tpr_train, _ = roc_curve(y_train_bal, y_train_predict_proba_xg)
roc_auc_train = auc(fpr_train, tpr_train)

# ROC curve for test data
fpr_test, tpr_test, _ = roc_curve(y_test_XGBC, y_predict_proba_xg)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC curve
plt.figure()
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC curve (area = {roc_auc_train:.2f})')
plt.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Test ROC curve (area = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
