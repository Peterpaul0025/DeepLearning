# MLP Neural Network from Scratch using NumPy to classify Iris dataset
# -----------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Part 1: Data Preparation --------------------
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y)

# Split into train (70%), validation (15%), test (15%)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.30, random_state=42, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)

# Normalize features using z-score
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# -------------------- Part 2: Neural Network Implementation --------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    expZ = np.exp(z - np.max(z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def initialize_parameters():
    np.random.seed(42)
    params = {
        'W1': np.random.randn(4, 8) * np.sqrt(1 / 4),
        'b1': np.zeros((1, 8)),
        'W2': np.random.randn(8, 6) * np.sqrt(1 / 8),
        'b2': np.zeros((1, 6)),
        'W3': np.random.randn(6, 3) * np.sqrt(1 / 6),
        'b3': np.zeros((1, 3))
    }
    return params

def forward_propagation(X, params):
    Z1 = X.dot(params['W1']) + params['b1']
    A1 = sigmoid(Z1)
    Z2 = A1.dot(params['W2']) + params['b2']
    A2 = sigmoid(Z2)
    Z3 = A2.dot(params['W3']) + params['b3']
    A3 = softmax(Z3)
    cache = (Z1, A1, Z2, A2, Z3, A3)
    return A3, cache

def compute_loss(Y_pred, Y_true):
    m = Y_true.shape[0]
    log_likelihood = -np.log(Y_pred[range(m), np.argmax(Y_true, axis=1)] + 1e-9)
    return np.sum(log_likelihood) / m

def backward_propagation(X, Y, params, cache):
    Z1, A1, Z2, A2, Z3, A3 = cache
    m = X.shape[0]

    dZ3 = A3 - Y
    dW3 = (A2.T).dot(dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = dZ3.dot(params['W3'].T)
    dZ2 = dA2 * sigmoid_derivative(Z2)
    dW2 = (A1.T).dot(dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2.dot(params['W2'].T)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = (X.T).dot(dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}
    return grads

def update_parameters(params, grads, lr):
    for key in params.keys():
        if key.startswith('W') or key.startswith('b'):
            params[key] -= lr * grads['d' + key]
    return params

# -------------------- Training --------------------
def get_batches(X, Y, batch_size):
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    for i in range(0, m, batch_size):
        idx = indices[i:i + batch_size]
        yield X[idx], Y[idx]

def train(X_train, Y_train, X_val, Y_val, lr=0.01, epochs=1000, batch_size=16):
    params = initialize_parameters()
    train_losses, val_accuracies = [], []
    best_val_acc, patience = 0, 0

    for epoch in range(epochs):
        for X_batch, Y_batch in get_batches(X_train, Y_train, batch_size):
            Y_pred, cache = forward_propagation(X_batch, params)
            grads = backward_propagation(X_batch, Y_batch, params, cache)
            params = update_parameters(params, grads, lr)

        if epoch % 50 == 0:
            Y_pred_train, _ = forward_propagation(X_train, params)
            train_loss = compute_loss(Y_pred_train, Y_train)
            Y_pred_val, _ = forward_propagation(X_val, params)
            val_preds = np.argmax(Y_pred_val, axis=1)
            val_labels = np.argmax(Y_val, axis=1)
            val_acc = np.mean(val_preds == val_labels)
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val_Acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
            else:
                patience += 1
            if patience > 10:
                print("Early stopping triggered.")
                break
    return params

params = train(X_train, Y_train, X_val, Y_val)

# -------------------- Evaluation --------------------
Y_pred_test, _ = forward_propagation(X_test, params)
y_pred_labels = np.argmax(Y_pred_test, axis=1)
y_true_labels = np.argmax(Y_test, axis=1)

print("\nTest Accuracy:", accuracy_score(y_true_labels, y_pred_labels))
print("Confusion Matrix:\n", confusion_matrix(y_true_labels, y_pred_labels))
print("\nClassification Report:\n", classification_report(y_true_labels, y_pred_labels, target_names=iris.target_names))

# -------------------- Decision Boundary Visualization --------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Inverse PCA transform grid back to original feature space
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_original = pca.inverse_transform(grid_points)
grid_scaled = scaler.transform(grid_original)
probs, _ = forward_propagation(grid_scaled, params)
Z = np.argmax(probs, axis=1).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=[iris.target_names[i] for i in y_true_labels], s=60, edgecolor='k')
plt.title("Decision Boundaries (PCA Projection)")
plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
plt.show()
