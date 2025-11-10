from import_package import *
# Load the Dataset
df_train = pd.read_csv('churn-bigml-80.csv')  # Loading the training dataset
df_test = pd.read_csv('churn-bigml-20.csv')  # Loading the testing dataset

# Combine the training and testing datasets for preprocessing or unified analysis
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# Dropping the 'State' column
df = df.drop(columns=['State'])

# Display the resulting dataframe
df
# show first lines
df.head()
#show last lines
df.tail()
# descriptive statistics
df.describe()
# show data types and missing values
df.info()
print("Data types of each column:")
print(df.dtypes)
# Show numeric columns
numeric_columns=df.select_dtypes(include=[np.number]).columns
numeric_columns
# View variable types of columns numerique
df[numeric_columns].dtypes
#  Show categorical columns
categorical_columns= df.select_dtypes(include=['object']).columns
# Count Unique Values (For each categorical variable, you can count the number of unique values)
for col in categorical_columns:
    print(f"Count of unique values in {col}: {df[col].nunique()}")
    #Value Counts for Categorical Variables:( This shows the frequency of each unique value in categorical columns.)
for col in categorical_columns:
    print(f"Value counts for {col}:\n{df[col].value_counts()}\n")
# shape of the dataset
print("Shape of the dataset (rows, columns):", df.shape)
# Histogram for each numeric column

df[numeric_columns].hist(bins=15, figsize=(15, 10))
plt.suptitle('Histograms of Numeric Columns')
plt.show()
# Pair plot to explore relationships between variables
sns.pairplot(df[numeric_columns])
plt.title('Pair Plot of Numeric Columns')
plt.show()
# Heatmap of correlations between columns
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of correlations between columns')
plt.show()
# Show missing values
missing_value=df.isnull().sum()
print(missing_value)
# View duplicate values
duplicated_columns=df.duplicated(subset=None,keep='first')
print(duplicated_columns)
# Outliers Detection
#method IQR(inter quantile Range)
Q1=df[numeric_columns].quantile(0.25)
Q3=df[numeric_columns].quantile(0.75)
IQR=Q3 - Q1
outliers=((df[numeric_columns]<Q1 -1.5*IQR)| (df[numeric_columns]>Q3+1.5*IQR)).sum()
print(outliers)
# Boxplot to view outliers

colors = sns.color_palette("Set3", n_colors=len(numeric_columns))
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(4, 5, i)
    sns.boxplot(y=df[col], color=colors[i-1])
    plt.title(f'Box Plot de {col}')
plt.tight_layout()
plt.show()


from explore_pretraitement_Dataset import numeric_columns,df
# # Define a function to manage outliers with IQR verification
def handle_outliers(data, column):
    if data[column].isnull().all():
        print(f"column {column} contains only NaN. Ignorée.")
        return  # Ignore empty columns
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        print(f"IQR for {column} is 0. Ignorée.")
        return  # Ignore if no variation
    min_value = Q1 - 1.5 * IQR
    max_value = Q3 + 1.5 * IQR
    # Clip the values to the calculated range
    data[column] = data[column].clip(lower=min_value, upper=max_value)

# Apply Outlier Processing
for column in numeric_columns:
    handle_outliers(df, column)
#treat missing values by dropping rows with any missing values
df=df.dropna()
print(df.isna().sum())
#  transformation of values from categorical to Numerique
df_encoder=pd.get_dummies(df,columns=['International plan','Voice mail plan'])
df=df_encoder
df
# Transform Boolean variables to int
bool_columns=df.select_dtypes(include=['bool']).columns
df[bool_columns] = df[bool_columns].astype(int)
df
#Normalization of the dataset
scaler=MinMaxScaler()
df_normalisés=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
df=df_normalisés
df
# Feature Selection
target_column='Churn'
X_train=df.drop(columns=[target_column])
y=df['Churn']
# Initialize the SelectKBest feature selection method
# chi2: Statistical test to assess the dependence between categorical target and features
# k=17: Select the top 17 features based on the chi-squared scores
selector = SelectKBest(chi2, k=24)

# Apply the feature selection to the training data
# X_train: Features in the training set
# y: Target variable
X_new_train = selector.fit_transform(X_train, y)

# Get the indices of the selected features
selected_indices = selector.get_support(indices=True)

# Retrieve the column names corresponding to the selected features
selected_columns = X_train.columns[selected_indices]

# Print the list of selected feature names
print("Selected Features:", selected_columns.tolist())





