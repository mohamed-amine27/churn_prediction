from import_package import *
# Split the data into training and test sets

X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_new_train, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes in the training set
smote_dt = SMOTE(random_state=42)
X_train_s, y_train_s = smote_dt.fit_resample(X_train_dt, y_train_dt)

# Initialize the decision tree classifier with default hyperparameters
model = DecisionTreeClassifier(
    random_state=42,  # Ensures reproducibility
    max_depth=5,  # Maximum depth of the tree
    min_samples_leaf=2,  # Minimum samples required at a leaf node
    min_samples_split=5,# Minimum samples required to split an internal node
    class_weight='balanced',
    min_impurity_decrease=0.02,  # Minimum impurity decrease required for a split
    max_leaf_nodes=50,  # Maximum number of leaf nodes
    criterion="entropy"  # Splitting criterion (entropy for information gain)
)

# Fit the model on the balanced training data
model.fit(X_train_s, y_train_s)

# Perform grid search to optimize hyperparameters
param_grid = {
    'max_depth': [3, 5, 7, 10],  # Test different maximum depths
    'min_samples_leaf': [1, 5, 10],  # Test different minimum leaf sizes
    'min_samples_split': [2, 5, 10, 20],  # Test different minimum splits
    'criterion': ['gini', 'entropy']  # Test different splitting criteria
}
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),  # Decision tree as the base model
    param_grid,  # Parameter grid for tuning
    cv=5,  # 5-fold cross-validation
    scoring='accuracy'  # Evaluate based on accuracy
)
# grid_search.fit(X_train_s, y_train_s)

# # Display the best parameters from the grid search
# print("Best Parameters:", grid_search.best_params_)

# Predict outcomes on the test set
y_pred_dt = model.predict(X_test_dt)
# Classification report for training data
y_train_predict_dt = model.predict(X_train_s)
y_train_predict_proba_dt = model.predict_proba(X_train_s)[:, 1]
classification_report_train_dt = classification_report(y_train_s, y_train_predict_dt)
# ROC curve for training data
fpr_train_dt, tpr_train_dt, _ = roc_curve(y_train_s, y_train_predict_proba_dt)
roc_auc_train_dt = auc(fpr_train_dt, tpr_train_dt)

# ROC curve for test data
fpr_test_dt, tpr_test_dt, _ = roc_curve(y_test_dt,  model.predict_proba(X_test_dt)[:, 1])
roc_auc_test_dt = auc(fpr_test_dt, tpr_test_dt)

# Plot ROC curve
plt.figure()
plt.plot(fpr_train_dt, tpr_train_dt, color='blue', lw=2, label=f'Train ROC curve (area = {roc_auc_train_dt:.2f})')
plt.plot(fpr_test_dt, tpr_test_dt, color='red', lw=2, label=f'Test ROC curve (area = {roc_auc_test_dt:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
accuracy = accuracy_score(y_test_dt, y_pred_dt)  # Calculate accuracy
print(f"Accuracy: {accuracy:.4f}")  # Print accuracy with four decimal places
print("Confusion Matrix:")

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_dt, y_pred_dt), display_labels=["Negative Class", "Positive Class"])
disp.plot(cmap="Blues")  # Use a blue colormap for the confusion matrix plot
print("Classification Report:")
class_report_dt=classification_report(y_test_dt, y_pred_dt)  # Print detailed classification metrics

# Calculate the ROC AUC Score
roc_score_dt = roc_auc_score(y_test_dt, model.predict_proba(X_test_dt)[:, 1])  # Compute AUC-ROC score for the test set
roc_auc_score_train_dt = roc_auc_score(y_train_s, y_train_predict_proba_dt)
print("ROC AUC SCORE TRAIN", roc_auc_score_train_dt)


print("ROC AUC Score:", roc_score_dt)
print("class Report(Test)")
print(class_report_dt)
print("Classification Report (Train):")
print(classification_report_train_dt)

# Visualize the decision tree
plt.figure(figsize=(20, 10))  # Set the figure size
tree.plot_tree(
    model,
    feature_names=selected_columns,  # Use the selected features for visualization
    class_names=['Non-Churn', 'Churn'],  # Class labels for the tree
    filled=True  # Color the nodes based on class probabilities
)
plt.title("Decision Tree")  # Set the title of the plot
plt.show()  # Display the plot
