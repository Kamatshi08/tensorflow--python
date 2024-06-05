import tensorflow as tf
import numpy as np

# Generate synthetic data for training
np.random.seed(42)
X_train = np.random.randn(100, 2).astype(np.float32)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.float32).reshape(-1, 1)

# Define model parameters
W = tf.Variable(tf.random.normal([2, 1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")

# Define the logistic regression model
def logistic_regression(x):
    return tf.sigmoid(tf.matmul(x, W) + b)

# Define the loss function (Binary Cross-Entropy)
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.losses.binary_crossentropy(y_true, y_pred))

# Define the optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.1)

# Training step function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = logistic_regression(x)
        loss = loss_fn(y, y_pred)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    return loss

# Training loop
epochs = 1000
for epoch in range(epochs):
    loss = train_step(X_train, y_train)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.numpy():.4f}, W = {W.numpy().flatten()}, b = {b.numpy().flatten()}")

# Predict
X_test = np.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
y_pred = logistic_regression(X_test)
print("Predicted probabilities:", y_pred.numpy())
print("Predicted classes:", (y_pred.numpy() > 0.5).astype(np.int32))
