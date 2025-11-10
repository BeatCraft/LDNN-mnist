import numpy as np

# ===== ユーティリティ =====
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    # z>0 のとき 1, それ以外 0
    return (z > 0).astype(z.dtype)

def softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)  # 数値安定化
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(a, y_onehot):
    # バッチ平均のクロスエントロピー
    return -np.mean(np.sum(y_onehot * np.log(a + 1e-12), axis=1))

# ===== ダミーデータ（バッチ=5, 入力=4, 隠れ=6, 出力=3）=====
np.random.seed(0)
m, n_in, n_hidden, n_out = 5, 4, 6, 3
A0 = np.random.randn(m, n_in)                             # 入力/前層出力: (m, n_in)
Y  = np.eye(n_out)[np.random.randint(0, n_out, size=m)]   # one-hot ラベル: (m, n_out)

# ===== パラメータ初期化 =====
W1 = 0.1 * np.random.randn(n_in, n_hidden)   # (n_in, n_hidden)
b1 = np.zeros((1, n_hidden))                 # (1, n_hidden)
W2 = 0.1 * np.random.randn(n_hidden, n_out)  # (n_hidden, n_out)
b2 = np.zeros((1, n_out))                    # (1, n_out)

# ===== 順伝播 =====
Z1 = A0 @ W1 + b1          # (m, n_hidden)
A1 = relu(Z1)              # (m, n_hidden)

Z2 = A1 @ W2 + b2          # (m, n_out)
A2 = softmax(Z2)           # (m, n_out)

loss = cross_entropy(A2, Y)

print("=== Forward ===")
print("A0:", A0.shape, "W1:", W1.shape, "b1:", b1.shape, "Z1:", Z1.shape, "A1:", A1.shape)
print("W2:", W2.shape, "b2:", b2.shape, "Z2:", Z2.shape, "A2:", A2.shape)
print("loss:", loss)

# ===== 逆伝播 =====
# 出力層（Softmax+CE）：δ2 = (A2 - Y) / m
delta2 = (A2 - Y) / m                   # (m, n_out)

# 勾配：dW2 = A1^T @ δ2,  db2 = sum(δ2)
dW2 = A1.T @ delta2                      # (n_hidden, n_out)  ← .T で (n_hidden, m) に
db2 = np.sum(delta2, axis=0, keepdims=True)  # (1, n_out)

# 隠れ層：δ1 = (δ2 @ W2^T) * ReLU'(Z1)
delta1 = (delta2 @ W2.T) * relu_derivative(Z1)  # (m, n_hidden)

# 勾配：dW1 = A0^T @ δ1,  db1 = sum(δ1)
dW1 = A0.T @ delta1                       # (n_in, n_hidden)
db1 = np.sum(delta1, axis=0, keepdims=True)     # (1, n_hidden)

print("\n=== Backward ===")
print("delta2:", delta2.shape, "dW2:", dW2.shape, "db2:", db2.shape)
print("delta1:", delta1.shape, "dW1:", dW1.shape, "db1:", db1.shape)

# ===== パラメータ更新（SGDの例）=====
lr = 0.1
W2 -= lr * dW2
b2 -= lr * db2
W1 -= lr * dW1
b1 -= lr * db1

print("\n=== After one SGD step ===")
print("W1[0]:", W1[0])
print("b1:", b1)
print("W2[0]:", W2[0])
print("b2:", b2)


