# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"D:\VS Code\Machine Learning\Classification Models\Logistic Regression\logit classification.csv")

# Dependent variable and Independent variable
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

#confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# This is to get the Models Accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

# Bias score
bias = classifier.score(x_train, y_train)
print(bias)

# Variance score
variance = classifier.score(x_test, y_test)
print(variance)


from sklearn.metrics import roc_curve, roc_auc_score
# Get predicted probabilities for the positive class (usually class 1)
y_prob = classifier.predict_proba(x_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

# Print AUC
print("AUC Score:", auc_score)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid()
plt.show()

 # ---------------------- Future Predicts -----------------------------------------

# Load the new data
new_data = pd.read_csv(r"D:\VS Code\Machine Learning\Classification Models\Logistic Regression\final1.csv")

# Extract the features
x_new = new_data.iloc[:, [3, 4]].values

# Feature Scaling
x_new_scaled = sc.transform(x_new)

# Predicting the results
y_new_pred = classifier.predict(x_new_scaled)

# Add predictions to the DataFrame
new_data['Predicted_Class'] = y_new_pred

# Save results to a new CSV
new_data.to_csv(r"D:\VS Code\Machine Learning\Classification Models\Logistic Regression\new_predictions.csv", index=False)