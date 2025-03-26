# XOR Logic Gate with TensorFlow (I use gpt for generating this README file)

This project implements a simple neural network using TensorFlow to learn the XOR logic gate. The model is built using `tf.Module` and trained with gradient descent.

## 📌 Overview
This script:
- Defines a **LogicGate** model using `tf.Module`.
- Trains the model using **gradient descent**.
- Evaluates the model on the **XOR truth table**.

---

## 🚀 Implementation Details

### 1️⃣ Define the `LogicGate` Model
The `LogicGate` class initializes weights and biases only once when called with input data.

```python
import tensorflow as tf

class LogicGate(tf.Module):
    def __init__(self):
        super().__init__()
        self.built = False

    def __call__(self, x, train=True):
        if not self.built:
            input_dim = x.shape[-1]
            self.w = tf.Variable(tf.random.normal([input_dim, 1]), name="weights")
            self.b = tf.Variable(tf.zeros([1]), name="bias")
            self.built = True

        z = tf.add(tf.matmul(x, self.w), self.b)
        return tf.sigmoid(z)
```

### 2️⃣ Define the Loss Function
The loss function computes the **Mean Squared Error (MSE)** between predicted and actual outputs.

```python
def compute_loss(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))
```

### 3️⃣ Training Function
The model is trained using **Gradient Descent**. The weights and biases are updated using gradients.

```python
def train_model(model, x_train, y_train, learning_rate=0.5, epochs=5000):
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model(x_train)
            loss = compute_loss(y_pred, y_train)
        
        grads = tape.gradient(loss, model.variables)
        for g, v in zip(grads, model.variables):
            v.assign_sub(learning_rate * g)
        
        if epoch % 1000 == 0:
            acc = compute_accuracy(model, x_train, y_train)
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}, Accuracy: {acc:.4f}")
```

### 4️⃣ Compute Accuracy
Rounding the predictions and comparing them with actual values to compute accuracy.

```python
def compute_accuracy(model, x, y_true):
    y_pred = model(x, train=False)
    y_pred_rounded = tf.round(y_pred)
    correct = tf.equal(y_pred_rounded, y_true)
    return tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()
```

---

## 📊 XOR Dataset

The dataset represents the XOR truth table:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 1       | 0       | 1      |
| 0       | 1       | 1      |
| 1       | 1       | 0      |

```python
import numpy as np

xor_table = np.array([[0, 0, 0],
                      [1, 0, 1],
                      [0, 1, 1],
                      [1, 1, 0]], dtype=np.float32)

x_train = xor_table[:, :2]
y_train = xor_table[:, 2:]
```

---

## 🔥 Training the Model

```python
model = LogicGate()
train_model(model, x_train, y_train)
```

---

## 🏆 Learned Parameters
After training, we extract the learned weights and bias:

```python
w1, w2 = model.w.numpy().flatten()
b = model.b.numpy().flatten()[0]
print(f"\nLearned weight for w1: {w1}")
print(f"Learned weight for w2: {w2}")
print(f"Learned bias: {b}\n")
```

---

## ✅ Model Predictions
We check the model's predictions against the XOR truth table.

```python
y_pred = model(x_train, train=False).numpy().round().astype(np.uint8)
print("Predicted Truth Table:")
print(np.column_stack((xor_table[:, :2], y_pred)))
```

---

## 🛠 Requirements
- `numpy`
- `tensorflow`

Install dependencies using:
```sh
pip install numpy tensorflow
```

---

## 📌 Conclusion
Since a single-layer perceptron cannot learn the XOR function, the model struggles to converge. For better results, use a **multi-layer perceptron (MLP)** with a hidden layer.

---

## 💡 Next Steps
- Modify the model to use a **hidden layer** (MLP)
- Experiment with **ReLU activation** instead of sigmoid
- Use **cross-entropy loss** instead of MSE for better classification results
