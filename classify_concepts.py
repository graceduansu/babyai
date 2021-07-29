x = np.vstack(activations)
y = sum(blocks, [])
y = np.array([INDEX[b] if b in INDEX else 0 for b in y])

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import StandardScaler


p = np.random.permutation(len(x))
N = int(len(p) / 5)
X_train, y_train = x[p[:N]], y[p[:N]]
X_test, y_test = x[p[N:]], y[p[N:]]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Decision Tree classifer object
clf = LogisticRegression()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

