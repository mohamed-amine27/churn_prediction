from import_package import *
# Split the data into training and testing sets
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_new_train, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance in the training data
smote_rf = SMOTE(random_state=42)
X_train_resample_rf, y_train_resample_rf = smote_rf.fit_resample(X_train_rf, y_train_rf)

# Initialize the Random Forest classifier with specified hyperparameters
model_rf = RandomForestClassifier(
    bootstrap=True,  # Use bootstrap sampling for training
    max_depth=5,  # Maximum depth of the trees
    min_samples_leaf=5,  # Minimum number of samples required at a leaf node
    min_samples_split=20,  # Minimum number of samples required to split an internal node
    max_leaf_nodes=50,  # Maximum number of leaf nodes
    n_estimators=50,  # Number of trees in the forest
    min_impurity_decrease=0.001,  # Minimum impurity decrease required for a split
    random_state=42  # Random state for reproducibility
)

# Define the grid of hyperparameters for GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100],  # Number of trees to test
    'max_depth': [3, 5, 7, 10],  # Different tree depths to test
    'min_samples_leaf': [1, 5, 10],  # Minimum samples per leaf to test
    'min_samples_split': [2, 5, 10, 20],  # Minimum samples to split a node
    'max_features': ['sqrt', 'log2'],  # Number of features considered for the best split
    'bootstrap': [True],  # Use bootstrap samples for building trees
}

# Perform grid search to find the best hyperparameters
grid_search_rf = GridSearchCV(
    estimator=model_rf,  # Base Random Forest model
    param_grid=param_grid_rf,  # Parameter grid to search
    cv=3,  # 3-fold cross-validation
    scoring='recall',  # Optimize the model for recall
    verbose=1,  # Display progress of the search
    n_jobs=-1  # Use all available processors
)
# grid_search_rf.fit(X_train_resample_rf, y_train_resample_rf)

# # Display the best parameters found by GridSearchCV
# print("Best parameters:", grid_search_rf.best_params_)

# Train the Random Forest model on the resampled training data
model_rf.fit(X_train_resample_rf, y_train_resample_rf)

# Get the best model from GridSearchCV
# best_model_rf = grid_search_rf.best_estimator_

# Predict the test set outcomes
y_pred_rf = model_rf.predict(X_test_rf)
y_train_predict_rf = model_rf.predict(X_train_resample_rf)
y_train_predict_proba_rf = model_rf.predict_proba(X_train_resample_rf)[:, 1]
classification_report_train_rf = classification_report(y_train_resample_rf, y_train_predict_rf)

# ROC curve for training data
fpr_train_rf, tpr_train_rf, _ = roc_curve(y_train_resample_rf, y_train_predict_proba_rf)
roc_auc_train_rf = auc(fpr_train_rf, tpr_train_rf)

# ROC curve for test data
fpr_test_rf, tpr_test_rf, _ = roc_curve(y_test_rf, model_rf.predict_proba(X_test_rf)[:, 1])
roc_auc_test_rf = auc(fpr_test_rf, tpr_test_rf)





# Evaluate the model's accuracy
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
print(f"Accuracy: {accuracy_rf:.4f}")  # Print accuracy with four decimal places

# Display the confusion matrix
print("Confusion Matrix:")
conf_matrix_rf = confusion_matrix(y_test_rf, y_pred_rf)  # Generate the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf, display_labels=["Negative Class", "Positive Class"])
disp.plot(cmap="Blues")  # Use a blue colormap for the confusion matrix plot

# Print classification report
print("Classification Report:")
class_report_rf=classification_report(y_test_rf, y_pred_rf)
print(class_report_rf)  # Detailed classification metrics
print("Classification Report (Train):")
print(classification_report_train_rf)

# Calculate and display the ROC AUC score
roc_score_rf = roc_auc_score(y_test_rf, model_rf.predict_proba(X_test_rf)[:, 1])  # Compute AUC-ROC score

roc_auc_score_train_rf = roc_auc_score(y_train_resample_rf, y_train_predict_proba_rf)
print("ROC AUC SCORE TRAIN", roc_auc_score_train_rf)
print("ROC AUC Score:", roc_score_rf)

# Plot ROC curve
plt.figure()
plt.plot(fpr_train_rf, tpr_train_rf, color='blue', lw=2, label=f'Train ROC curve (area = {roc_auc_train_rf:.2f})')
plt.plot(fpr_test_rf, tpr_test_rf, color='red', lw=2, label=f'Test ROC curve (area = {roc_auc_test_rf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
