from import_package import *
# Split the dataset into training and testing sets
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_new_train, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
# smote_svm = SMOTETomek(random_state=42)
# X_res, y_res = smote_svm.fit_resample(X_train_svm, y_train_svm)

# Initialize the Support Vector Machine (SVM) model with specified hyperparameters
svm_model = SVC(
    C=165,  # Regularization parameter
    degree=3,  # Degree of the polynomial kernel (only relevant for 'poly' kernel)
    gamma=1,  # Kernel coefficient
    kernel='poly'  # Use polynomial kernel
)

# Train the SVM model on the resampled training data
svm_model.fit(X_train_svm, y_train_svm)

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'C': [1, 10, 100],  # Regularization strength
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel types
    'gamma': [0.01, 0.1, 1, 'scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    'degree': [2, 3, 4, 5]  # Degree of polynomial kernel
}

# # Configure GridSearchCV for hyperparameter optimization
# grid_search = GridSearchCV(
#     estimator=svm_model,  # SVM model to tune
#     param_grid=param_grid,  # Hyperparameter grid
#     cv=5,  # 5-fold cross-validation
#     scoring='accuracy',  # Scoring metric
#     verbose=1,  # Verbosity level for progress updates
#     n_jobs=-1  # Use all available processors
# )

# # Perform the hyperparameter search
# grid_search.fit(X_res, y_res)

# # Display the best hyperparameters found by GridSearchCV
# print("Best hyperparameters found:", grid_search.best_params_)

# Make predictions on the test set using the trained SVM model
y_pred_svm = svm_model.predict(X_test_svm)
y_train_predict_SVM= svm_model.predict(X_train_svm)
classification_report_train_SVM = classification_report(y_train_svm, y_train_predict_SVM)


# Evaluate the performance of the model
accuracy_svm = accuracy_score(y_test_svm, y_pred_svm)  # Compute accuracy score
matrix_svm = confusion_matrix(y_test_svm, y_pred_svm) # Generate the confusion matrix
class_report_svm=classification_report(y_test_svm, y_pred_svm)

# Display evaluation results
print("Accuracy:", accuracy_svm)
print("\nClassification report:\n",class_report_svm)
print("Classification Report (Train):")
print(classification_report_train_SVM)

# Visualize the confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=matrix_svm,
    display_labels=["Negative Class", "Positive Class"]  # Class labels
)
disp.plot(cmap="Blues")  # Use a blue colormap for the plot
