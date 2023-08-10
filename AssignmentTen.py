import rpy2.robjects as robjects
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


# Load the Titanic dataset in R and convert it into an R dataframe
r = robjects.r
r('titanic_data <- read.csv("C:/Users/king/Desktop/jupyterythonassignment/Python_Assignment10/titanic/titanic.csv")')

# Fetch the Titanic dataset from R to Python
titanic_data = robjects.globalenv['titanic_data']

# Convert the R dataframe to a Pandas dataframe
df = pd.DataFrame(dict(titanic_data.items()))

# Print the head of the dataset
result_head = r('head(titanic_data)')
print(result_head)

#EDA Task 2
# print("Summary statistics of numerical features:")
print(df.describe())

# Check the data types and missing values
print("Data types and missing values:")
print(df.info())

# Visualize the distribution of numerical features using histograms
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=df, x=feature, kde=True, bins=20, color='skyblue')
    plt.title(f'Histogram of {feature}')
plt.tight_layout()
plt.show()


# Plot the bar plot for Passenger Class (Pclass) distribution and save it as "barplot.png"
r('png("barplot.png")')
r('barplot(table(titanic_data$Pclass), main="Passenger Class Distribution", xlab="Passenger Class", ylab="Count", col="orange")')
r['dev.off']()

# Load and display the saved bar plot using matplotlib in Python
img = plt.imread("barplot.png")
plt.imshow(img)
plt.axis("off")
plt.show()


# Preprocessing 3
# Handling Missing Values
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Feature Engineering
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Encoding Categorical Variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

# Drop unnecessary columns for modeling
X = df.drop(columns=['PassengerId', 'Name', 'Survived', 'Cabin', 'Ticket', 'Title'])
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement Logistic Regression
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)
y_pred_logreg = logreg_model.predict(X_test)

##Model 1: Implement Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

#Model 2: Evaluate the models
print("Logistic Regression Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Classification Report:")
print(classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))

print("\nDecision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# Scatter Plot for 'Age' vs 'Fare'
plt.scatter(df['Age'], df['Fare'], c=df['Survived'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Scatter Plot: Age vs Fare (Color by Survival)')
plt.colorbar(label='Survived')
plt.show()

# Model 3: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict using the Random Forest model
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_confusion_matrix = confusion_matrix(y_test, y_pred_rf)

print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Confusion Matrix:")
print(rf_confusion_matrix)


#Visualization:import seaborn as sns

# Select only numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=[float, int])

# Compute the correlation matrix
corr_matrix = numeric_df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
