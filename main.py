import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier


def decision_tree_classification(df, dependentVar):
    # Separate features (X) and target variable (y)
    X = df.loc[:, df.columns != dependentVar]
    y = df[dependentVar].values

    # One-hot encode categorical variables
    X = pd.get_dummies(X)

    # Standardize features
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)

    # Train Decision Tree model
    dtModel = DecisionTreeClassifier()
    dtModel.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = dtModel.predict(X_test)

    # Evaluation metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    recall = recall_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass

    # Return results as a dictionary
    results = {
        'confusion_matrix': cm,
        'accuracy_score': accuracy,
        'precision_score': precision,
        'recall_score': recall,
        'f1_score': f1
    }

    return results


def naive_bayes_classification(df, dependentVar):
    # Separate features (X) and target variable (y)
    X = df.loc[:, df.columns != dependentVar]
    y = df[dependentVar].values

    # One-hot encode categorical variables
    X = pd.get_dummies(X)

    # Standardize features
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)

    # Train Gaussian Naive Bayes model
    nbModel = GaussianNB()
    nbModel.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = nbModel.predict(X_test)

    # Evaluation metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    recall = recall_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass

    # Return results as a dictionary
    results = {
        'confusion_matrix': cm,
        'accuracy_score': accuracy,
        'precision_score': precision,
        'recall_score': recall,
        'f1_score': f1
    }

    return results


# Example Usage
df = pd.read_csv('./cereal.csv')
dependentVar = 'protein'
classification_results = naive_bayes_classification(df, dependentVar)
print(" Naive Bayes Classification Results:")
print(classification_results)

print("-----\n")


df1 = pd.read_csv('./cereal.csv')
dependentVar = 'protein'
classification_results_dt = decision_tree_classification(df1, dependentVar)
print("Decision Tree Classification Results:")
print(classification_results_dt)