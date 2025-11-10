import pandas as pd  # Used for data manipulation and analysis
import numpy as np  # Provides support for numerical computations and array operations

import matplotlib.pyplot as plt  # Used for creating visualizations such as line charts, histograms, etc.
import seaborn as sns  # Provides advanced visualizations and easier styling for statistical data plots

from sklearn import tree  # Provides tools for working with decision trees, such as visualization and implementation

from sklearn.preprocessing import MinMaxScaler  # Scales features to a specific range (default: 0 to 1)
from sklearn.feature_selection import chi2, SelectKBest  # Used for feature selection based on statistical tests

from sklearn.linear_model import LogisticRegression  # Implements Logistic Regression, a simple linear model for classification
from sklearn.tree import DecisionTreeClassifier  # Implements Decision Tree Classifier for classification tasks
from sklearn.ensemble import RandomForestClassifier  # Implements Random Forest, an ensemble model based on decision trees
from sklearn.neighbors import KNeighborsClassifier  # Implements k-Nearest Neighbors for classification
from sklearn.svm import SVC  # Implements Support Vector Classifier for classification
from sklearn.ensemble import AdaBoostClassifier  # Implements AdaBoost, an ensemble model that uses boosting
from sklearn.ensemble import GradientBoostingClassifier  # Implements Gradient Boosting, another ensemble boosting model
from xgboost import XGBClassifier  # Implements eXtreme Gradient Boosting, an optimized boosting algorithm

from sklearn.model_selection import train_test_split, GridSearchCV  # Splits data into training/testing and performs hyperparameter tuning

from imblearn.over_sampling import SMOTE  # Handles class imbalance by oversampling the minority class using SMOTE
from imblearn.combine import SMOTEENN  # Combines SMOTE and Edited Nearest Neighbors to balance data
from imblearn.combine import SMOTETomek  # Combines SMOTE and Tomek links for oversampling and cleaning

from sklearn.metrics import (  # Provides performance metrics for model evaluation
    accuracy_score,  # Calculates the accuracy of the predictions
    classification_report,  # Provides a detailed report of precision, recall, F1-score, etc.
    confusion_matrix,  # Generates a confusion matrix for classification evaluation
    roc_auc_score,  # Calculates the ROC-AUC score for binary classification tasks
    ConfusionMatrixDisplay  # Visualizes the confusion matrix
)
# ROC curve for both training and testing data
from sklearn.metrics import roc_curve, auc

import joblib  # Used for saving and loading trained models to/from disk
from IPython.display import FileLink  # Creates downloadable links for files in Jupyter/IPython notebooks
