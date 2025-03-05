# Simple Neural Network - No Libraries, Just Basics

# Sigmoid activation function
def sigmoid(x):
    # Approximate e as 2.71828
    return 1 / (1 + 2.71828 ** (-x))

# Derivative of sigmoid function
def sigmoid_deriv(x):
    return x * (1 - x)

# Input values and target outputs
inputs = [0.05, 0.10]  # i1, i2
targets = [0.01, 0.99]  # Expected o1, o2

# Initialize weights and biases
w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60
lr = 0.5  # Learning rate

# Number of iterations (epochs)
epochs = 1  # Can be increased later

# Training loop
for i in range(epochs):
    # Forward pass
    h1 = sigmoid(w1 * inputs[0] + w2 * inputs[1] + b1)
    h2 = sigmoid(w3 * inputs[0] + w4 * inputs[1] + b1)
    
    o1 = sigmoid(w5 * h1 + w6 * h2 + b2)
    o2 = sigmoid(w7 * h1 + w8 * h2 + b2)
    
    # Calculate error
    err1 = 0.5 * (targets[0] - o1) ** 2
    err2 = 0.5 * (targets[1] - o2) ** 2
    total_err = err1 + err2
    
    if i == 0:
        print(f"Starting error: {total_err}")
        print(f"o1 = {o1}, o2 = {o2}")
    
    # Backpropagation - Output layer
    d_o1 = (o1 - targets[0]) * sigmoid_deriv(o1)
    d_o2 = (o2 - targets[1]) * sigmoid_deriv(o2)
    
    w5 -= lr * d_o1 * h1
    w6 -= lr * d_o1 * h2
    w7 -= lr * d_o2 * h1
    w8 -= lr * d_o2 * h2
    b2 -= lr * (d_o1 + d_o2)
    
    # Backpropagation - Hidden layer
    d_h1 = (d_o1 * w5 + d_o2 * w7) * sigmoid_deriv(h1)
    d_h2 = (d_o1 * w6 + d_o2 * w8) * sigmoid_deriv(h2)
    
    w1 -= lr * d_h1 * inputs[0]
    w2 -= lr * d_h1 * inputs[1]
    w3 -= lr * d_h2 * inputs[0]
    w4 -= lr * d_h2 * inputs[1]
    b1 -= lr * (d_h1 + d_h2)
    
    # Forward pass again to check improvement
    h1 = sigmoid(w1 * inputs[0] + w2 * inputs[1] + b1)
    h2 = sigmoid(w3 * inputs[0] + w4 * inputs[1] + b1)
    o1 = sigmoid(w5 * h1 + w6 * h2 + b2)
    o2 = sigmoid(w7 * h1 + w8 * h2 + b2)
    
    # Calculate new error
    err1 = 0.5 * (targets[0] - o1) ** 2
    err2 = 0.5 * (targets[1] - o2) ** 2
    new_err = err1 + err2
    
    print(f"New error: {new_err}")
    print(f"New o1 = {o1}, o2 = {o2}")
    print(f"Weights: w1={w1}, w2={w2}, w3={w3}, w4={w4}")
    print(f"         w5={w5}, w6={w6}, w7={w7}, w8={w8}")
    print(f"Biases: b1={b1}, b2={b2}")
