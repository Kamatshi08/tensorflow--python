import tensorflow as tf
import numpy as np

# Generate synthetic data for training
np.random.seed(42)
X_train = np.random.rand(100).astype(np.float32)
y_train = 2 * X_train + 1 + np.random.normal(0, 0.1, 100).astype(np.float32)

# Define model parameters
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Define the linear model
def linear_model(x):
    return W * x + b

# Define the loss function (Mean Squared Error)
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Define the optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.1)

# Training step function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = linear_model(x)
        loss = loss_fn(y, y_pred)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    return loss

# Training loop
epochs = 100
for epoch in range(epochs):
    loss = train_step(X_train, y_train)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.numpy():.4f}, W = {W.numpy():.4f}, b = {b.numpy():.4f}")

# Predict
X_test = np.array([0.0, 0.5, 1.0], dtype=np.float32)
y_pred = linear_model(X_test)
print("Predicted values:", y_pred.numpy())
