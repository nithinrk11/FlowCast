!pip install tensorflow scikit-learn
!git clone https://github.com/nithinrk11/FlowCast.git
import numpy as np

#decision tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

#random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

#XGboost
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

#XGboost Hyperparameter
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

#DNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

#Models
dec = DecisionTreeClassifier()
rf = RandomForestClassifier()
xg = XGBClassifier()
best_xgb = XGBClassifier()
model = Sequential()



# Load the noisy dataset
data = pd.read_csv('/content/FlowCast/noisy_crowd_data2.csv', parse_dates=['Timestamp'])

# Encode the 'Crowd_Type' column
label_encoder = LabelEncoder()
data['Crowd_Type_Label'] = label_encoder.fit_transform(data['Crowd_Type'])

# Define features and target
X = data[['Noisy_Crowd_Count']]
y = data['Crowd_Type_Label']


# Decision tree
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the decision tree model
dec = DecisionTreeClassifier(random_state=42)

# Train the model
dec.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dec.predict(X_test)

# Calculate metrics
#accuracy = accuracy_score(y_test, y_pred)
#conf_matrix = confusion_matrix(y_test, y_pred)
#class_report = classification_report(y_test, y_pred)

#print(f"Decision tree Accuracy: {accuracy:.2f}")
#print("\nDecision tree Confusion Matrix:")
#print(conf_matrix)
#print("\nDecision tree Classification Report:")
#print(class_report)
#-----------------------------------------------------------------------------#

# random forest
# Binarize the output for multiclass classification
y_bin = label_binarize(y, classes=np.unique(y))


# Create a Random Forest classifier
rf = RandomForestClassifier(n_estimators=50, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf.predict(X_test)

# Evaluate accuracy
#accuracy_rf = accuracy_score(y_test, y_pred_rf)
#print(f"Random Forest Accuracy: {accuracy_rf:.2f}")

# Confusion Matrix
#conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
#print("\nRandom Forest Confusion Matrix:")
#print(conf_matrix_rf)

# Classification Report
#class_report_rf = classification_report(y_test, y_pred_rf)
#print("\nRandom Forest Classification Report:")
#print(class_report_rf)
#----------------------------------------------------------------------------#
# XGBoost
# Set the number of classes explicitly
num_classes = len(np.unique(y))

# Create an XGBoost classifier with num_class parameter
xg = XGBClassifier(num_class=num_classes)

# Train the model
xg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xg.predict(X_test)

# Evaluate accuracy
#accuracy = accuracy_score(y_test, y_pred)
#print(f"XGBoost Accuracy: {accuracy:.2f}")

# Confusion Matrix
#conf_matrix = confusion_matrix(y_test, y_pred)
#print("\nXGBoost Confusion Matrix:")
#print(conf_matrix)

# Classification Report
#class_report = classification_report(y_test, y_pred)
#print("\nXGBoost Classification Report:")
#print(class_report)
#---------------------------------------------------------------------------#

# XGBoost Hyperparameter
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# Create an XGBoost classifier
xgb = XGBClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Use the best model for prediction
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)

# Evaluate accuracy
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Tuned XGBoost Accuracy with Best Model: {accuracy:.2f}")

# Confusion Matrix
#conf_matrix = confusion_matrix(y_test, y_pred)
#print("\nTuned XGBoost Confusion Matrix:")
#print(conf_matrix)

# Classification Report
#class_report = classification_report(y_test, y_pred)
#print("\nTuned XGBoost Classification Report:")
#print(class_report)
#----------------------------------------------------------------------------#

# DNN
# Set random seed for NumPy
np.random.seed(42)

# Convert labels to one-hot encoding
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Build the DNN model

model.add(Dense(128, input_dim=1, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Adjusted output layer for three classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_one_hot, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test_one_hot, axis=1)

# Calculate metrics
#accuracy = accuracy_score(y_test_labels, y_pred_labels)
#conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
#class_report = classification_report(y_test_labels, y_pred_labels)

#print(f"DNN Accuracy: {accuracy:.2f}")
#print("\nDNN Confusion Matrix:")
#print(conf_matrix)
#print("\nDNN Classification Report:")
#print(class_report)

#Dumping the models into correspondind pickle file in write mode
pickle.dump(dec,open('dec.pkl','wb'))
pickle.dump(rf,open('rf.pkl','wb'))
pickle.dump(xg,open('xg.pkl','wb'))
pickle.dump(best_xgb,open('best_xgb.pkl','wb'))
pickle.dump(model,open('model.pkl','wb'))



