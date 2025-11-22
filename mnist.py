import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import time

# ==========================
# 0. LOAD & PREPROCESS MNIST
# ==========================

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

print("Shape dữ liệu gốc:", X.shape)

# Chuẩn hóa về [0,1]
X = X / 255.0

# Lấy subset cho nhanh
n_samples = 20000
X = X[:n_samples]
y = y[:n_samples]
print("Dùng subset:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)
plt.rcParams['figure.figsize'] = (6, 4)

# ==========================
# 1. PCA + EIGENVALUES
# ==========================

max_components = 200
pca = PCA(n_components=max_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

eigenvalues = pca.explained_variance_
explained_var_ratio = pca.explained_variance_ratio_

print("\n=== PCA INFO ===")
print("Eigenvalues (10 đầu):", eigenvalues[:10])
print("Explained variance ratio (10 đầu):", explained_var_ratio[:10])

# ==========================
# 2. EXPLAINED VARIANCE PLOTS
# ==========================

cum_explained = np.cumsum(explained_var_ratio)

plt.figure()
plt.plot(range(1, max_components + 1), explained_var_ratio)
plt.xlabel("PC index")
plt.ylabel("Explained variance ratio")
plt.title("Explained Variance per Component")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(range(1, max_components + 1), cum_explained)
plt.xlabel("PC index")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative Explained Variance")
plt.grid(True)
plt.ylim(0, 1.05)
plt.show()

# ==========================
# 3. VISUALIZE PCA COMPONENTS (HEATMAP + COLORBAR)
# ==========================

n_show = 16   # số PC đầu tiên để đưa vào báo cáo
components = pca.components_[:n_show]

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle("16 Principal Components đầu tiên (Heatmap 28×28)", fontsize=14)

# Min/max chung cho colorbar
vmin = components.min()
vmax = components.max()

for i in range(n_show):
    ax = axes[i // 4, i % 4]
    im = ax.imshow(components[i].reshape(28, 28),
                   cmap='seismic', vmin=vmin, vmax=vmax)
    ax.set_title(f"PC{i+1}")
    ax.axis("off")

# Colorbar chung
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
cbar.set_label("Component Weight")

plt.show()

# ==========================
# 4. SCATTER PLOT 2D & 3D
# ==========================

idx = np.random.choice(len(X_train_pca), size=5000, replace=False)
X2 = X_train_pca[idx, :2]
y2 = y_train[idx]

plt.figure(figsize=(6, 5))
scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y2, s=5, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Scatter Plot (PC1 vs PC2)")
cbar = plt.colorbar(scatter)
cbar.set_label("Digit label")
plt.show()

X3 = X_train_pca[idx, :3]

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=y2, s=5, alpha=0.7)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
fig.colorbar(p, label="Digit label")
ax.set_title("PCA 3D Scatter Plot")
plt.show()

# ==========================
# 5. RECONSTRUCTION
# ==========================

def reconstruct_from_pca(X_pca_full, pca_obj, k):
    W_k = pca_obj.components_[:k, :]
    Z_k = X_pca_full[:, :k]
    return Z_k @ W_k + pca_obj.mean_

n_vis = 10
X_original = X_test[:n_vis]
Xp = X_test_pca[:n_vis]

X_rec_10 = reconstruct_from_pca(Xp, pca, 10)
X_rec_50 = reconstruct_from_pca(Xp, pca, 50)
X_rec_100 = reconstruct_from_pca(Xp, pca, 100)

def show_rec(X_orig, X10, X50, X100, n=5):
    plt.figure(figsize=(10, 8))
    for i in range(n):
        plt.subplot(4, n, i+1)
        plt.imshow(X_orig[i].reshape(28, 28), cmap='gray')
        plt.title("Gốc"); plt.axis("off")

        plt.subplot(4, n, i+1+n)
        plt.imshow(X10[i].reshape(28, 28), cmap='gray')
        plt.title("10D"); plt.axis("off")

        plt.subplot(4, n, i+1+2*n)
        plt.imshow(X50[i].reshape(28, 28), cmap='gray')
        plt.title("50D"); plt.axis("off")

        plt.subplot(4, n, i+1+3*n)
        plt.imshow(X100[i].reshape(28, 28), cmap='gray')
        plt.title("100D"); plt.axis("off")

    plt.suptitle("So sánh ảnh gốc và tái tạo PCA", fontsize=14)
    plt.show()

show_rec(X_original, X_rec_10, X_rec_50, X_rec_100, n=5)

# ==========================
# 6. RECONSTRUCTION ERROR vs K
# ==========================

n_eval = 2000
X_eval = X_test[:n_eval]
Xp_eval = X_test_pca[:n_eval]

k_list = [5, 10, 20, 30, 50, 70, 100, 150, 200]
mse_list = []

print("\n=== Reconstruction MSE ===")
for k in k_list:
    X_rec_k = reconstruct_from_pca(Xp_eval, pca, k)
    mse = mean_squared_error(X_eval, X_rec_k)
    mse_list.append(mse)
    print(f"k={k:3d}  →  MSE = {mse:.6f}")

plt.figure()
plt.plot(k_list, mse_list, marker='o')
plt.xlabel("Số lượng PC (k)")
plt.ylabel("MSE Reconstruction")
plt.title("Ảnh hưởng của k lên chất lượng tái tạo")
plt.grid(True)
plt.show()

# ==========================
# 7. CLASSIFIER RAW vs PCA
# ==========================

def train_and_eval_logreg(Xtr, ytr, Xte, yte, desc=""):
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    t1 = time.time()
    acc = clf.score(Xte, yte)
    print(f"[{desc}] Accuracy = {acc:.4f}, Training time = {t1 - t0:.2f}s")
    return acc, t1 - t0

print("\n=== Logistic Regression Performance ===")

acc_raw, t_raw = train_and_eval_logreg(
    X_train, y_train, X_test, y_test, "Raw 784D"
)

k_cls = 50
acc_pca, t_pca = train_and_eval_logreg(
    X_train_pca[:, :k_cls], y_train,
    X_test_pca[:, :k_cls], y_test,
    f"PCA {k_cls}D"
)

print("\n=== SUMMARY ===")
print(f"Raw 784D:  acc={acc_raw:.4f}, time={t_raw:.2f}s")
print(f"PCA {k_cls}D:   acc={acc_pca:.4f}, time={t_pca:.2f}s")
