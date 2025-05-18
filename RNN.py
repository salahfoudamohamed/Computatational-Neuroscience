

# words
corpus = ["I", "love", "deep", "learning"]
vocab = corpus
vocab_size = len(vocab)

word_to_ix = {"I": 0, "love": 1, "deep": 2, "learning": 3}
ix_to_word = {0: "I", 1: "love", 2: "deep", 3: "learning"}

def one_hot(index, size):
    return [1 if i == index else 0 for i in range(size)]


def mat_vec_mul(matrix, vector):
    result = []
    for row in matrix:
        dot = 0
        for a, b in zip(row, vector):
            dot += a * b
        result.append(dot)
    return result


def vec_add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]


def tanh(x_list):
    def approx_tanh(x):
        return x - (x**3)/6 + (x**5)/120
    return [approx_tanh(x) for x in x_list]


def tanh_deriv(vec):
    return [1 - x**2 for x in vec]


def softmax(vec):
    def approx_exp(x):
        return 1 + x + (x**2)/2 + (x**3)/6
    exps = [approx_exp(x) for x in vec]
    total = sum(exps)
    return [e / total for e in exps]

# Fixed small weights 
Wxh = [
    [0.1, 0.2, 0.3, 0.4],
    [0.2, 0.1, 0.0, -0.1],
    [0.0, 0.3, -0.2, 0.1]
]  

Whh = [
    [0.1, 0.0, 0.0],
    [0.0, 0.1, 0.0],
    [0.0, 0.0, 0.1]
] 

Why = [
    [0.1, 0.2, 0.3],
    [0.0, 0.1, 0.0],
    [0.2, -0.1, 0.1],
    [0.1, 0.0, -0.2]
]  

bh = [0, 0, 0]
by = [0, 0, 0, 0]


lr = 0.05

# Training loop
for epoch in range(500):
    inputs = [one_hot(word_to_ix[word], vocab_size) for word in corpus[:3]]
    target_index = word_to_ix[corpus[3]]

    # Forward pass
    h = [0, 0, 0]
    hs = []
    xs = []

    for x in inputs:
        xh = mat_vec_mul(Wxh, x)
        hh = mat_vec_mul(Whh, h)
        total = vec_add(vec_add(xh, hh), bh)
        h = tanh(total)
        hs.append(h)
        xs.append(x)

    y = vec_add(mat_vec_mul(Why, h), by)
    probs = softmax(y)

    # Loss 
    if epoch % 100 == 0:
        pred = ix_to_word[probs.index(max(probs))]
        print(f"Epoch {epoch}, predicted: {pred}")

    # Backward pass

    # dL/dy (from softmax + cross-entropy)
    dy = [p for p in probs]
    dy[target_index] -= 1

    # Gradients 
    dWhy = [[0 for _ in range(len(h))] for _ in range(vocab_size)]
    dby = [0 for _ in range(vocab_size)]

    for i in range(vocab_size):
        for j in range(len(h)):
            dWhy[i][j] += dy[i] * h[j]
        dby[i] += dy[i]

    # dL/dh
    dh = [0 for _ in range(len(h))]
    for j in range(len(h)):
        for i in range(vocab_size):
            dh[j] += dy[i] * Why[i][j]

    # Gradients for Wxh, Whh, bh 
    dWxh = [[0 for _ in range(vocab_size)] for _ in range(len(h))]
    dWhh = [[0 for _ in range(len(h))] for _ in range(len(h))]
    dbh = [0 for _ in range(len(h))]

    for t in reversed(range(3)):
        h_t = hs[t]
        x_t = xs[t]
        h_prev = hs[t - 1] if t > 0 else [0 for _ in range(len(h))]

        dh_raw = [dh_i * (1 - h_i ** 2) for dh_i, h_i in zip(dh, h_t)]

        for i in range(len(h)):
            for j in range(vocab_size):
                dWxh[i][j] += dh_raw[i] * x_t[j]
            for j in range(len(h)):
                dWhh[i][j] += dh_raw[i] * h_prev[j]
            dbh[i] += dh_raw[i]

        # Backprop previous hidden
        dh_new = [0 for _ in range(len(h))]
        for j in range(len(h)):
            for i in range(len(h)):
                dh_new[j] += dh_raw[i] * Whh[i][j]
        dh = dh_new

    # Update weights 
    for i in range(vocab_size):
        for j in range(len(h)):
            Why[i][j] -= lr * dWhy[i][j]
        by[i] -= lr * dby[i]

    for i in range(len(h)):
        for j in range(vocab_size):
            Wxh[i][j] -= lr * dWxh[i][j]
        for j in range(len(h)):
            Whh[i][j] -= lr * dWhh[i][j]
        bh[i] -= lr * dbh[i]

 # prediction 
print("\nAfter training:")
h = [0, 0, 0]
for x in inputs:
    xh = mat_vec_mul(Wxh, x)
    hh = mat_vec_mul(Whh, h)
    total = vec_add(vec_add(xh, hh), bh)
    h = tanh(total)

y = vec_add(mat_vec_mul(Why, h), by)
probs = softmax(y)
predicted_index = probs.index(max(probs))
print("Predicted word:", ix_to_word[predicted_index])
