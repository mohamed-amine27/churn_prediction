from import_package import *
# Split the dataset into training and testing sets
X_train_ABC, X_test_ABC, y_train_ABC, y_test_ABC = train_test_split(X_new_train, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTEENN (combination of SMOTE and Edited Nearest Neighbors)
smote = SMOTE(random_state=42)
X_train_resample_ABC, y_train_resample_ABC = smote.fit_resample(X_train_ABC, y_train_ABC)

# Initialize AdaBoostClassifier with a DecisionTreeClassifier as the base estimator
model_ABC = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),  # Decision tree base model
    random_state=42,  # Seed for random number generator
    n_estimators=50,  # Number of boosting iterations
    learning_rate=0.017,  # Learning rate for the model
    algorithm='SAMME'  # The algorithm used for boosting
)

# Train the AdaBoost model on the resampled training data
model_ABC.fit(X_train_resample_ABC, y_train_resample_ABC)

# Define the hyperparameter grid for GridSearchCV to optimize AdaBoost model
param_grid = {
    'n_estimators': [50, 100, 200, 300],  # Number of boosting rounds (trees)
    'learning_rate': [0.01, 0.1, 0.5, 1.0, 1.5],  # Step size for boosting
    'algorithm': ['SAMME', 'SAMME.R'],  # Type of boosting algorithm
    'estimator': [
        # Different configurations of DecisionTreeClassifier as the base estimator
        DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, criterion='gini'),
        DecisionTreeClassifier(max_depth=3, min_samples_split=15, min_samples_leaf=10, criterion='entropy')
    ]
}

# Configure GridSearchCV for hyperparameter optimization
# grid_search = GridSearchCV(estimator=model_ABC, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

# # # Perform the grid search to find the best hyperparameters
# grid_search.fit(X_train_resample_ABC, y_train_resample_ABC)

# # # Display the best hyperparameters found by GridSearchCV
# print("Best hyperparameters:", grid_search.best_params_)

# Make predictions on the test set using the trained AdaBoost model
y_pred_ABC = model_ABC.predict(X_test_ABC)
y_train_predict_proba_ABC = model_ABC.predict_proba(X_train_resample_ABC)[:, 1]
roc_auc_score_train_ABC = roc_auc_score(y_train_resample_ABC, y_train_predict_proba_ABC)
print("ROC AUC SCORE TRAIN", roc_auc_score_train_ABC)

# Calculate the accuracy of the model
accuracy_ABC = accuracy_score(y_test_ABC, y_pred_ABC)
print(f"Accuracy: {accuracy_ABC:.4f}")

# Generate and display the confusion matrix
print("Confusion Matrix:")
conf_matrix_abc = confusion_matrix(y_test_ABC, y_pred_ABC)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_abc, display_labels=["Negative Class", "Positive Class"])
disp.plot(cmap="Blues")

# Print the classification report (precision, recall, f1-score)
print("Classification Report:")
class_report_ABC=classification_report(y_test_ABC, y_pred_ABC)
print(class_report_ABC)

# Calculate the ROC AUC score to evaluate model performance
roc_score_ABC = roc_auc_score(y_test_ABC, model_ABC.predict_proba(X_test_ABC)[:, 1])
print("roc_auc_score (optimized):", roc_score_ABC)
