import numpy as np


def softmax(x):
    # subtract max for numerical stability
    exp_x = np.exp(
        x - np.max(x, axis=-1, keepdims=True)
    )

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


def calculate_attention(X):
    # X shape: (number of words, embedding size)
    d_model = X.shape[1]
    d_k = 64

    # fixed seed so the scores are the same on every run
    np.random.seed(42)

    # random (untrained) projection matrices
    WQ = np.random.randn(d_model, d_k)
    WK = np.random.randn(d_model, d_k)
    WV = np.random.randn(d_model, d_k)

    # Query, Key, Value
    Q = X @ WQ
    K = X @ WK
    V = X @ WV

    # how much each word matches every other word
    scores = Q @ K.T

    # scaled dot-product attention
    scaled_scores = scores / np.sqrt(d_k)

    # each row becomes probabilities that sum to 1
    attention_weights = softmax(scaled_scores)

    # average attention each word receives
    word_scores = attention_weights.mean(axis=0)

    return word_scores
