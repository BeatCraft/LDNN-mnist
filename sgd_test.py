import numpy as np

# ----- ユーティリティ -----
def softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)  # 数値安定化
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(a, y_onehot):
    # バッチ平均のクロスエントロピー
    return -np.mean(np.sum(y_onehot * np.log(a + 1e-9), axis=1))

# ----- ダミーデータ（バッチ=5, 入力=4, クラス=3）-----
np.random.seed(0)
m, n_in, n_out = 5, 4, 3
A_prev = np.random.randn(m, n_in)     # 前層の出力（=この層の入力）: (m, n_in)
Y = np.eye(n_out)[np.random.randint(0, n_out, size=m)]  # one-hot ラベル: (m, n_out)

# ----- パラメータ初期化 -----
W = 0.1 * np.random.randn(n_in, n_out)  # (n_in, n_out)
b = np.zeros((1, n_out))                # (1, n_out)

# ===== 順伝播 =====
Z = A_prev @ W + b            # (m, n_out)
A = softmax(Z)                # (m, n_out)
loss = cross_entropy(A, Y)

print("Forward shapes:")
print(" A_prev:", A_prev.shape, " W:", W.shape, " b:", b.shape)
print(" Z:", Z.shape, " A(softmax):", A.shape)
print(" loss:", loss)

# ===== 逆伝播（勾配計算）=====
# Softmax + CE の基本式： delta = A - Y  （各サンプルの誤差）
delta = (A - Y) / m           # バッチ平均をここで反映（/m）

# dW = A_prev^T @ delta,  db = sum(delta)
dW = A_prev.T @ delta         # (n_in, n_out)  ← .T で (n_in, m) にして行列積
db = np.sum(delta, axis=0, keepdims=True)  # (1, n_out)

print("\nBackward shapes:")
print(" delta:", delta.shape, " dW:", dW.shape, " db:", db.shape)

# ===== 更新（SGDの例）=====
lr = 0.1
W -= lr * dW
b -= lr * db

print("\nUpdated params:")
print(" W[0]:", W[0])
print(" b:", b)


