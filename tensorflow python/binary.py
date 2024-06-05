import tensorflow as tf

# Generate synthetic data for binary classification
X_train = tf.random.normal([1000, 2])
y_train = tf.cast(X_train[:, 0]**2 + X_train[:, 1]**2 < 1, tf.float32)
y_train = tf.reshape(y_train, [-1, 1])

# Define the neural network model
class SimpleNN(tf.Module):
    def __init__(self):
        # Initialize weights and biases for 2 layers
        self.W1 = tf.Variable(tf.random.normal([2, 4]), name="W1")
        self.b1 = tf.Variable(tf.zeros([4]), name="b1")
        self.W2 = tf.Variable(tf.random.normal([4, 1]), name="W2")
        self.b2 = tf.Variable(tf.zeros([1]), name="b2")

    def __call__(self, x):
        # Forward pass
        h1 = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        y = tf.sigmoid(tf.matmul(h1, self.W2) + self.b2)
        return y

# Instantiate the model
model = SimpleNN()

# Define the loss function (Binary Cross-Entropy)
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

# Define the optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.01)

# Training step function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
epochs = 1000
for epoch in range(epochs):
    loss = train_step(model, X_train, y_train)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.numpy():.4f}")

# Predict
X_test = tf.constant([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [-1.0, -1.0]], dtype=tf.float32)
y_pred = model(X_test)
print("Predicted probabilities:", y_pred.numpy())
print("Predicted classes:", tf.cast(y_pred > 0.5, tf.int32).numpy())
