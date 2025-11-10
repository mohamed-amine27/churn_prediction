from import_package import *
# Split the dataset into training and testing sets
X_train_gbm, X_test_gbm, y_train_gbm, y_test_gbm = train_test_split(X_new_train, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote_gbm = SMOTE(random_state=42)
X_train_gbm_balanced, y_train_gbm_balanced = smote_gbm.fit_resample(X_train_gbm, y_train_gbm)

# Initialize the Gradient Boosting Model (GBM) with specified hyperparameters
gbm = GradientBoostingClassifier(
    n_estimators=17,  # Number of boosting stages (trees)
    learning_rate=0.088,  # Step size for each boosting step
    max_depth=4,  # Maximum depth of the trees
     
    random_state=42  # Seed for reproducibility
)

# Train the GBM model on the balanced training data
gbm.fit(X_train_gbm_balanced, y_train_gbm_balanced)

# Define the hyperparameter grid for GridSearchCV to optimize the GradientBoostingClassifier
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of boosting rounds (trees)
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate for each boosting step
    'max_depth': [3, 5, 7],  # Maximum depth of the individual trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be a leaf node
}

# Perform grid search to find the best hyperparameters for the GBM
# grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, 
#                            cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Display the best hyperparameters found by GridSearchCV
# print("Best Parameters:", grid_search.best_params_)

# Make predictions on the test set using the trained GBM model
y_predict_gbm = gbm.predict(X_test_gbm)
y_predict_proba_gbm = gbm.predict_proba(X_test_gbm)[:, 1]  # Probabilities for the positive class

# Evaluate the model using accuracy, confusion matrix, and ROC AUC score
conf_matrix_gbm = confusion_matrix(y_test_gbm, y_predict_gbm)
accuracy_gbm = accuracy_score(y_test_gbm, y_predict_gbm)
roc_auc_score_gbm = roc_auc_score(y_test_gbm, y_predict_proba_gbm)
classification_report_gbm = classification_report(y_test_gbm, y_predict_gbm)
y_train_predict_proba_gbm = gbm.predict_proba(X_train_gbm_balanced)[:, 1]
roc_auc_score_train_gbm= roc_auc_score(y_train_gbm_balanced, y_train_predict_proba_gbm)
print("ROC AUC SCORE TRAIN", roc_auc_score_train_gbm)

# Print the evaluation results
print("accuracy_gbm = ", accuracy_gbm)
print("conf_matrix_gbm = ", conf_matrix_gbm)

# Display the confusion matrix as a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_gbm, display_labels=["Negative Class", "Positive Class"])
disp.plot(cmap="Blues")

# Print the ROC AUC score
print("roc_auc_score_gbm = ", roc_auc_score_gbm)

# Print the classification report (precision, recall, f1-score)
print("Classification Report:")
print(classification_report_gbm)
y_train_predict_gbm= gbm.predict(X_train_gbm_balanced)
classification_report_train_gbm = classification_report(y_train_gbm_balanced, y_train_predict_gbm)
print("Classification Report (Train):")
print(classification_report_train_gbm)


# ROC curve for training data
fpr_train, tpr_train, _ = roc_curve(y_train_gbm_balanced, y_train_predict_proba_gbm)
roc_auc_train_gbm = auc(fpr_train, tpr_train)

# ROC curve for test data
fpr_test_gbm, tpr_test_gbm, _ = roc_curve(y_test_gbm, y_predict_proba_gbm)
roc_auc_test_gbm = auc(fpr_test_gbm, tpr_test_gbm)

# Plot ROC curve
# plt.figure()
# plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC curve (area = {roc_auc_train_gbm:.2f})')
# plt.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Test ROC curve (area = {roc_auc_test_gbm:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()

