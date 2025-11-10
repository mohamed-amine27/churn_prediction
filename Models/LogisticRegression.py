from import_package import *

# Split the data into training and test sets
# X_train_rl and y_train_rl are used to train the model
# X_test_rl and y_test_rl are used to test and evaluate the model
X_train_rl, X_test_rl, y_train_rl, y_test_rl = train_test_split(X_new_train, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTETomek
# Applies SMOTE oversampling to increase minority class samples and
# removes Tomek Links to clean overlapping samples
smote_tomek = SMOTE(random_state=42)
X_train_resample_rl, y_train_resample_rl = smote_tomek.fit_resample(X_train_rl, y_train_rl)

# Define the hyperparameter grid to test
# Parameters for optimizing the logistic regression model with GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength (inverse of regularization)
    'solver': ['liblinear', 'saga'],  # Optimization algorithms to test
    'penalty': ['l1', 'l2'],  # Regularization types to control model complexity
    'max_iter': [100, 200, 300]  # Maximum number of iterations for convergence
}

# Initialize the logistic regression model with specific parameters
logreg_model = LogisticRegression(random_state=42, max_iter=200, C=185, penalty='l2',class_weight='balanced',solver='liblinear')

# Train the logistic regression model on the resampled data
logreg_model.fit(X_train_resample_rl, y_train_resample_rl)
# Make predictions on the test set
y_pred_rl = logreg_model.predict(X_test_rl)  # Class predictions
y_pred_prob_rl = logreg_model.predict_proba(X_test_rl)[:, 1]  # Probabilities for the positive class
# Evaluate the model
accuracy_rl = accuracy_score(y_test_rl, y_pred_rl)  # Compute model accuracy
conf_matrix_rl = confusion_matrix(y_test_rl, y_pred_rl)  # Generate the confusion matrix
roc_auc_rl = roc_auc_score(y_test_rl, y_pred_prob_rl)  # Calculate the ROC-AUC score
y_train_predict_proba_rl = logreg_model.predict_proba(X_train_resample_rl)[:, 1]
roc_auc_score_train_rl = roc_auc_score(y_train_resample_rl, y_train_predict_proba_rl)
y_train_predict_rl = logreg_model.predict(X_train_resample_rl)
classification_report_train_rl = classification_report(y_train_resample_rl, y_train_predict_rl)

# ROC curve for training data
fpr_train_rl, tpr_train_rl, _ = roc_curve(y_train_resample_rl, y_train_predict_proba_rl)
roc_auc_train_rl = auc(fpr_train_rl, tpr_train_rl)

# ROC curve for test data
fpr_test_rl, tpr_test_rl, _ = roc_curve(y_test_rl, y_pred_prob_rl)
roc_auc_test_rl = auc(fpr_test_rl, tpr_test_rl)



class_report_rl = classification_report(y_test_rl, y_pred_rl)  # Detailed classification report
# Display the results
print(f"Accuracy: {accuracy_rl:.2f}")  # Print accuracy with two decimal places
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rl, display_labels=["Negative Class", "Positive Class"])
disp.plot(cmap="Blues")  # Display the confusion matrix as a plot with a blue colormap
print(f"Roc AUC Score: {roc_auc_rl:.2f}")  # Print the ROC-AUC score with two decimal places
print("ROC AUC SCORE TRAIN", roc_auc_score_train_rl)
print("Classification Report(Test):")  # Print the classification report
print(class_report_rl)
print("Classification Report (Train):")
print(classification_report_train_rl)

# Plot ROC curve
plt.figure()
plt.plot(fpr_train_rl, tpr_train_rl, color='blue', lw=2, label=f'Train ROC curve (area = {roc_auc_train_rl:.2f})')
plt.plot(fpr_test_rl, tpr_test_rl, color='red', lw=2, label=f'Test ROC curve (area = {roc_auc_test_rl:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

