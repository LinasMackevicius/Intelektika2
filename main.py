import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample data (replace with your actual data)
data = {
    "name": ["Raisin Bran", "Total Raisin Bran", "Special K", "Honey Nut Cheerios"],
    "calories": [120, 140, 110, 110],
    "protein": [3, 3, 6, 3],
    "fiber": [14, 14, 0, 11.5],
    "sugar": [12, 14, 0, 10.5],
    "sodium": [210, 190, 230, 250],
    "type": ["C", "C", "C", "C"]
}

# Convert data to pandas dataframe
df = pd.DataFrame(data)

# Define features and target variable
features = ["calories", "protein", "fiber", "sugar", "sodium"]
target = "type"

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2)

# Create a Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Train the classifier
clf.fit(X_train, y_train)

# Predict the type of a new cereal (replace with desired values)
new_cereal = {"calories": 130, "protein": 4, "fiber": 8, "sugar": 8, "sodium": 180}

# Predict the type (assuming "C" is the healthy category)
predicted_type = clf.predict([new_cereal])[0]

# Print the result (assuming "C" is the healthy category)
if predicted_type == "C":
    print("The new cereal is predicted to be a healthy option based on the given features.")
else:
    print("The new cereal is not predicted to be a healthy option based on the given features.")

# Note: This is a simplified example. A more comprehensive approach would involve:
# - Using a larger dataset representing the entire cereal market.
# - Considering additional features like vitamins, minerals, and ingredients.
# - Evaluating the performance of the model using appropriate metrics.
# - Consulting a healthcare professional for personalized dietary advice.
