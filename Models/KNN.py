from import_package import *
# Split the data into training and testing sets
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_new_train, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTETomek
smote_tomek = SMOTE(random_state=42)
X_train_resample_knn, y_train_resample_knn = smote_tomek.fit_resample(X_train_knn, y_train_knn)

# Display the class distribution after applying SMOTETomek
# print("Class distribution after SMOTETomek:")
# print(y_train_resample_knn.value_counts())



# Initialize the K-Nearest Neighbors (KNN) classifier with specified hyperparameters
knn = KNeighborsClassifier(
    algorithm='auto',  # Algorithm used for finding nearest neighbors
    leaf_size=25,  # Leaf size for tree-based algorithms
    metric='minkowski',  # Distance metric used for neighbors calculation
    n_neighbors=110,  # Number of neighbors to consider
    p=1,  # Use Manhattan distance if p=1, Euclidean distance if p=2
    weights='uniform'  # Weight points by the inverse of their distance
)

# Fit the KNN model on the resampled training data
knn.fit(X_train_resample_knn, y_train_resample_knn)

# Define the parameter grid for hyperparameter optimization
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],  # Different values for the number of neighbors
    'weights': ['uniform', 'distance'],  # Weight points uniformly or by distance
    'p': [1, 2],  # Test Manhattan and Euclidean distances
    'leaf_size': [20, 30, 40],  # Different leaf sizes for optimization
    'algorithm': ['auto', 'ball_tree', 'kd_tree']  # Algorithms for nearest neighbors search
}

# Uncomment the following lines to perform hyperparameter tuning with GridSearchCV
# grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
# grid_search.fit(X_train_resample_knn, y_train_resample_knn)

# # Display the best parameters found by GridSearchCV
# print("Best parameters: ", grid_search.best_params_)
# best_knn = grid_search.best_estimator_

# Make predictions on the test set
y_pred_knn = knn.predict(X_test_knn)
y_train_predict_proba_knn = knn.predict_proba(X_train_resample_knn)[:, 1]
roc_auc_score_train_knn = roc_auc_score(y_train_resample_knn, y_train_predict_proba_knn)
print("ROC AUC SCORE TRAIN", roc_auc_score_train_knn)
y_train_predict_knn = knn.predict(X_train_resample_knn)
classification_report_train_knn= classification_report(y_train_resample_knn, y_train_predict_knn)
 

#ROC curve for training data
fpr_train_knn, tpr_train_knn, _ = roc_curve(y_train_resample_knn, y_train_predict_proba_knn)
roc_auc_train_knn = auc(fpr_train_knn, tpr_train_knn)

# ROC curve for test data
fpr_test_knn, tpr_test_knn, _ = roc_curve(y_test_knn,  knn.predict_proba(X_test_knn)[:, 1])
roc_auc_test_knn = auc(fpr_test_knn, tpr_test_knn)



# Evaluate the model
accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)  # Compute accuracy score
roc_auc_knn = roc_auc_score(y_test_knn, knn.predict_proba(X_test_knn)[:, 1])  # Compute ROC AUC score
conf_matrix_knn = confusion_matrix(y_test_knn, y_pred_knn)  # Generate the confusion matrix

# Display the confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix_knn, 
    display_labels=["Negative Class", "Positive Class"]
)
disp.plot(cmap="Blues")  # Use a blue colormap for the plot

# Generate the classification report
class_report_knn = classification_report(y_test_knn, y_pred_knn)

# Display the evaluation metrics
print(f"Accuracy: {accuracy_knn:.2f}")  # Print accuracy with two decimal places
print(f"ROC AUC: {roc_auc_knn:.2f}")  # Print ROC AUC score with two decimal places
print("Confusion Matrix:")
print(conf_matrix_knn)  # Print the confusion matrix
print("Classification Report:")
print(class_report_knn)# Print the classification report
print("Classification Report (Train):")
print(classification_report_train_knn)
# Plot ROC curve
plt.figure()
plt.plot(fpr_train_knn, tpr_train_knn, color='blue', lw=2, label=f'Train ROC curve (area = {roc_auc_train_knn:.2f})')
plt.plot(fpr_test_knn, tpr_test_knn, color='red', lw=2, label=f'Test ROC curve (area = {roc_auc_test_knn:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
