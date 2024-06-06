from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load the Iris dataset
iris = datasets.load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize the SVM classifier with a linear kernel
clf = svm.SVC(kernel='linear')

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test)
