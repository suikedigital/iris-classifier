import argparse
import os
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

outputs_dir = os.path.join(os.path.dirname(__file__),"../outputs")
os.makedirs(outputs_dir, exist_ok=True)

def return_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_model(model):
    filename = f"{outputs_dir}/iris_model_{return_timestamp()}.joblib"
    joblib.dump(model, filename)

def save_results(output):
    filename = f"{outputs_dir}/iris_results_{return_timestamp()}.txt"

    with open(filename, "w") as f:
        f.write(output)

def create_and_save_confusion_matrix_image(Y_Test, Y_Pred):
    labels = ['Setosa', 'Versicolor', 'Virginica']
    cm = confusion_matrix(Y_Test, Y_Pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')

    filename = f"{outputs_dir}/confusion_matrix.png"
    plt.savefig(filename)
    plt.close()

def main(test_size, random_state):

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    Y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Create and train the Decision Tree model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = model.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(Y_test, Y_pred)

    # Generate a confusion matrix
    confusion = confusion_matrix(Y_test, Y_pred)

    # Create and train the K-NN model
    model2 = KNeighborsClassifier(n_neighbors=5)
    model2.fit(X_train, Y_train)

    # Make predictions with the K-NN model
    Y_pred2 = model2.predict(X_test)

    confusion2 = confusion_matrix(Y_test, Y_pred2)

    # Tuning the model (already 1.0 so unnecessary, but included for completeness)
    model3 = DecisionTreeClassifier(max_depth=1, random_state=42) # kept cycling down untill we got a result that was less than 1.0 for example of overtune.
    model3.fit(X_train, Y_train)
    Y_pred3 = model3.predict(X_test)

    confusion3 = confusion_matrix(Y_test, Y_pred3)
    

    output = f"""
Test Size: {test_size}
Random State: {random_state}

Decision Tree Accuracy: {accuracy}
Confusion Matrix:
{confusion}

K-NN Accuracy: {accuracy_score(Y_test, Y_pred2)}
K-NN Confusion Matrix:
{confusion2}

Tuned Model Accuracy: {accuracy_score(Y_test, Y_pred3)}
Tuned Model Confusion Matrix:
{confusion3}
    """

    save_results(output)
    save_model(model)
    # Create and save the final confusion matrix images
    create_and_save_confusion_matrix_image(Y_test, Y_pred)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris model with decision tree and K-nn")
    parser.add_argument("--test-size", type=float, default=0.2, help='Test set proportion')
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibilitiy")
    args = parser.parse_args()

    main(args.test_size, args.random_state)
