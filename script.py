import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import codecademylib3

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
print(transactions.info())

# How many fraudulent transactions?
print(transactions.isFraud.sum())
# Summary statistics on amount column

# Create isPayment field
transactions['isPayment'] = np.where(
    (transactions['type'] == "PAYMENT") | (transactions['type'] == "DEBIT"), 
    1, 0)

# Create isMovement field
transactions['isMovement'] = np.where(
  (
    (transactions['type'] == "CASH_OUT") |
    (transactions['type'] == "TRANSFER")
  ), 1, 0
)

# Create accountDiff field
transactions['accountDiff'] = abs(transactions['oldbalanceOrg'] - transactions['oldbalanceDest'])

# Create features and label variables
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]

label = transactions['isFraud']
# Split dataset
features_train, features_test, label_train, label_test = train_test_split(features, label, test_size=0.3)

# Normalize the features variables
scaler = StandardScaler()
scaler.fit_transform(features_train)
scaler.transform(features_test)

# Fit the model to the training data
lreg = LogisticRegression()
lreg.fit(features_train, label_train)

# Score the model on the training data
print(lreg.score(features_train, label_train))

# Score the model on the test data
print(lreg.score(features_test, label_test))

# Print the model coefficients
print(lreg.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
transaction4 = np.array([4000.22, 1.0, 0.0, 22222])

# Combine new transactions into a single array
sample_transactions = [transaction1, transaction2, transaction3, transaction4]

# Normalize the new transactions
sample_transactions = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
print(lreg.predict(sample_transactions))
print(lreg.predict_proba(sample_transactions))
# Show probabilities on the new transactions
