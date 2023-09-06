import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the dataset
df = pd.read_csv('Iris.csv')
# Encode the target variable
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
# Get the feature columns
# Take the last 15 records for future prediction
X_future = df.tail(15).drop(['Species'], axis=1)

# Split the remaining records into training and testing sets
X_train_test = df.head(len(df) - 15).drop(['Species'], axis=1)
y_train_test = df.head(len(df) - 15)['Species']
X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.3, random_state=42)

# Scale the features using min-max normalization
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_future = scaler.transform(X_future)

# Create a KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Apply K-fold cross-validation with 10 folds
knn_kfold = cross_val_score(knn, X_train_test, y_train_test, cv=10)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

# Calculate measuring scores
print("KNN K-fold cross-validation scores:\n", knn_kfold)
print("KNN Average fold cross-validation score:", knn_kfold.mean())
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print("KNN Recall:", recall_score(y_test, y_pred, average='weighted'))
print("KNN Precision:", precision_score(y_test, y_pred, average='weighted'))
print("KNN F1 score:", f1_score(y_test, y_pred, average='weighted'))

# Make predictions on the future data
future_pred = knn.predict(X_future)
print("KNN Predictions for the future data:")
print(le.inverse_transform(future_pred))

# Create Naive Bayes Model
nb = GaussianNB()

# Apply K-fold cross-validation with 10 folds
nb_kfold = cross_val_score(nb, X_train_test, y_train_test, cv=10)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = nb.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score on the testing data
print("\n---------------------------------------------")
print("\nNB K-fold cross-validation scores:\n", nb_kfold)
print("NB Average fold score:", nb_kfold.mean())
print("NB Accuracy:", accuracy_score(y_test, y_pred))
print("NB Precision:", precision_score(y_test, y_pred, average='weighted'))
print("NB Recall:", recall_score(y_test, y_pred, average='weighted'))
print("NB F1 score:", f1_score(y_test, y_pred, average='weighted'))

# Make predictions on the future data
future_pred = nb.predict(X_future)
print("Naive Bayes Predictions for the future data:")
print(le.inverse_transform(future_pred))

# Create ANN Model
ann = MLPClassifier(hidden_layer_sizes=(8, 4), activation='relu', max_iter=1000)

# Apply K-fold cross-validation with 10 folds
ann_kfold = cross_val_score(ann, X_train_test, y_train_test, cv=10)

# Fit the model to the training data
ann.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = ann.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score on the testing data
print("\n---------------------------------------------")
print("\nANN K-fold cross-validation scores:\n", ann_kfold)
print("ANN Average fold score:", ann_kfold.mean())
print("\nANN Accuracy:", accuracy_score(y_test, y_pred))
print("ANN Precision:", precision_score(y_test, y_pred, average='weighted'))
print("ANN Recall:", recall_score(y_test, y_pred, average='weighted'))
print("ANN F1 score:", f1_score(y_test, y_pred, average='weighted'))

# Make predictions on the future data
future_pred = ann.predict(X_future)
future_pred = le.inverse_transform(future_pred)
print("ANN Predictions for the future data:")
print(future_pred)