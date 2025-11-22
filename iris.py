# FULL COMPARISON: PCA vs Incremental PCA vs t-SNE (raw) on MNIST
# (Một cell duy nhất, trực quan hóa + các metric so sánh)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time

# ==========================
# 0. SETUP & LOAD MNIST
# ==========================

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (6, 4)

rng = np.random.RandomState(42)

print("=== LOAD MNIST ===")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_all = mnist.data.astype(np.float32)
y_all = mnist.target.astype(int)
print("Full MNIST shape:", X_all.shape)

# Chuẩn hóa về [0,1]
X_all /= 255.0

# Lấy 20k mẫu đầu cho nhất quán với phần PCA trước
n_total = 20000
X_all = X_all[:n_total]
y_all = y_all[:n_total]
print("Using subset:", X_all.shape)

# Tạo 1 subset chung 5000 mẫu cho toàn bộ so sánh
n_sub = 5000
idx_sub = rng.choice(n_total, size=n_sub, replace=False)
X_sub = X_all[idx_sub]
y_sub = y_all[idx_sub]

print("Common subset for comparison:", X_sub.shape)

# Helper: hàm tính stress (global structure)
def compute_stress(X_high, X_low, m=500):
    """
    Stress = ||D_high - D_low|| / ||D_high||
    X_high: dữ liệu trước giảm chiều
    X_low : embedding 2D
    m     : số điểm sử dụng (sample)
    """
    n = min(len(X_high), len(X_low), m)
    if n < 2:
        return np.nan
    idx = rng.choice(len(X_high), size=n, replace=False)
    D_high = pairwise_distances(X_high[idx])
    D_low = pairwise_distances(X_low[idx])
    return np.linalg.norm(D_high - D_low) / np.linalg.norm(D_high)

# Dictionary lưu mọi thứ theo từng phương pháp
methods = {}

# ==========================
# 1. PCA (50D) – BASELINE
# ==========================

print("\n=== PCA (50 components) ===")
t0 = time.time()
pca = PCA(n_components=50, random_state=42)
X_pca50 = pca.fit_transform(X_sub)
t_pca = time.time() - t0
print(f"PCA fit+transform time: {t_pca:.2f}s")

methods["PCA"] = {
    "X_high": X_sub,                 # 784D
    "X2": X_pca50[:, :2],            # cho scatter + metrics
    "X3": X_pca50[:, :3],
    "X_clf": X_pca50,                # cho classification (50D)
    "runtime": t_pca,
}

# ==========================
# 2. INCREMENTAL PCA (50D)
# ==========================

print("\n=== Incremental PCA (50 components) ===")
ipca = IncrementalPCA(n_components=50, batch_size=200)

t0 = time.time()
ipca.fit(X_sub)
X_ipca50 = ipca.transform(X_sub)
t_ipca = time.time() - t0
print(f"IPCA fit+transform time: {t_ipca:.2f}s")

methods["IPCA"] = {
    "X_high": X_sub,
    "X2": X_ipca50[:, :2],
    "X3": X_ipca50[:, :3],
    "X_clf": X_ipca50,
    "runtime": t_ipca,
}

# ==========================
# 3. t-SNE (RAW 2D & 3D)
# ==========================

print("\n=== t-SNE (raw 784D, 2D) ===")
tsne_raw_2d = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    n_iter=1000,
    random_state=42,
    verbose=0,
)
t0 = time.time()
X_tsne_raw2 = tsne_raw_2d.fit_transform(X_sub)
t_tsne_raw2 = time.time() - t0
print(f"t-SNE raw 2D time: {t_tsne_raw2:.2f}s")

print("\n=== t-SNE (raw 784D, 3D) ===")
tsne_raw_3d = TSNE(
    n_components=3,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    n_iter=1000,
    random_state=42,
    verbose=0,
)
t0 = time.time()
X_tsne_raw3 = tsne_raw_3d.fit_transform(X_sub)
t_tsne_raw3 = time.time() - t0
print(f"t-SNE raw 3D time: {t_tsne_raw3:.2f}s")

methods["t-SNE"] = {
    "X_high": X_sub,
    "X2": X_tsne_raw2,
    "X3": X_tsne_raw3,
    "X_clf": X_tsne_raw2,   # dùng 2D cho classification
    "runtime": t_tsne_raw2,
}

# ==========================
# 4. SCATTER 2D – GRID SO SÁNH
# ==========================

print("\n=== Plot 2D scatter for PCA / IPCA / t-SNE ===")

method_order_2d = ["PCA", "IPCA", "t-SNE"]

n_methods = len(method_order_2d)
n_cols = 3
n_rows = 1

plt.figure(figsize=(5 * n_cols, 4 * n_rows))
for i, name in enumerate(method_order_2d, start=1):
    info = methods[name]
    X2 = info["X2"]
    ax = plt.subplot(n_rows, n_cols, i)
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=y_sub, s=5, alpha=0.7, cmap="tab10")
    ax.set_title(name)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
plt.suptitle("2D Embeddings – PCA / IPCA / t-SNE (raw)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()

# ==========================
# 5. SCATTER 3D – PCA / IPCA / t-SNE
# ==========================

print("\n=== Plot 3D scatter for PCA / IPCA / t-SNE ===")

methods_3d = ["PCA", "IPCA", "t-SNE"]
n_methods3d = len(methods_3d)
n_cols = 3
n_rows = 1

fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
for i, name in enumerate(methods_3d, start=1):
    info = methods[name]
    X3 = info["X3"]
    ax = fig.add_subplot(n_rows, n_cols, i, projection="3d")
    p = ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2],
                   c=y_sub, s=5, alpha=0.7, cmap="tab10")
    ax.set_title(name)
    ax.set_xlabel("Comp1")
    ax.set_ylabel("Comp2")
    ax.set_zlabel("Comp3")

plt.suptitle("3D Embeddings – PCA / IPCA / t-SNE (raw)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()

# ==========================
# 6. t-SNE Perplexity Demo (5, 30, 100) – RAW
# ==========================

print("\n=== t-SNE Perplexity experiment (raw 784D) ===")
perplexities = [5, 30, 100]
n_tsne_perp = 2000
idx_perp = rng.choice(n_sub, size=n_tsne_perp, replace=False)
X_perp = X_sub[idx_perp]
y_perp = y_sub[idx_perp]

plt.figure(figsize=(15, 4))
for i, perp in enumerate(perplexities, start=1):
    print(f"  Running t-SNE (raw) with perplexity={perp}")
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",
        init="pca",
        n_iter=1000,
        random_state=42,
        verbose=0,
    )
    X_tsne_perp = tsne.fit_transform(X_perp)
    ax = plt.subplot(1, len(perplexities), i)
    sc = ax.scatter(X_tsne_perp[:, 0], X_tsne_perp[:, 1],
                    c=y_perp, s=5, alpha=0.7, cmap="tab10")
    ax.set_title(f"perplexity = {perp}")
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")

plt.suptitle("t-SNE (raw 784D) with Different Perplexities", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ==========================
# 7. METRICS: Silhouette, DBI, CHI, STRESS
# ==========================

print("\n=== METRICS: Silhouette, Davies–Bouldin, Calinski–Harabasz, Stress ===")

sil_scores = {}
dbi_scores = {}
chi_scores = {}
stress_scores = {}

for name, info in methods.items():
    X2 = info["X2"]
    print(f"\n>>> {name}")
    # Silhouette
    sil = silhouette_score(X2, y_sub)
    sil_scores[name] = sil
    print(f"  Silhouette (2D): {sil:.4f}")

    # DBI & CHI
    dbi = davies_bouldin_score(X2, y_sub)
    chi = calinski_harabasz_score(X2, y_sub)
    dbi_scores[name] = dbi
    chi_scores[name] = chi
    print(f"  Davies–Bouldin Index (lower better): {dbi:.4f}")
    print(f"  Calinski–Harabasz Index (higher better): {chi:.2f}")

    # Stress (global)
    X_high = info["X_high"]
    stress = compute_stress(X_high, X2, m=500)
    stress_scores[name] = stress
    print(f"  Stress (global structure, lower better): {stress:.4f}")

# Bar plots cho Silhouette
plt.figure(figsize=(7, 4))
plt.bar(sil_scores.keys(), sil_scores.values())
plt.xticks(rotation=0)
plt.ylabel("Silhouette (higher = better)")
plt.title("Silhouette Scores (2D embeddings)")
plt.tight_layout()
plt.show()

# Bar plots cho Stress
plt.figure(figsize=(7, 4))
plt.bar(stress_scores.keys(), stress_scores.values())
plt.xticks(rotation=0)
plt.ylabel("Stress (lower = better)")
plt.title("Global Structure Stress")
plt.tight_layout()
plt.show()

# ==========================
# 8. CLASSIFICATION trên EMBEDDINGS
# ==========================

print("\n=== Classification (Logistic Regression) on embeddings ===")

# Split chỉ số train/test trên subset 5000 mẫu
idx_all = np.arange(n_sub)
idx_train, idx_test = train_test_split(
    idx_all, test_size=0.2, stratify=y_sub, random_state=42
)

clf_acc = {}
clf_time = {}

def eval_logreg_on_embedding(X_emb, name):
    Xtr = X_emb[idx_train]
    Xte = X_emb[idx_test]
    ytr = y_sub[idx_train]
    yte = y_sub[idx_test]

    # scale để ổn định và dễ hội tụ
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    clf = LogisticRegression(max_iter=5000, n_jobs=-1)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    t1 = time.time()
    acc = clf.score(Xte, yte)
    clf_acc[name] = acc
    clf_time[name] = t1 - t0
    print(f"[{name:10s}] acc = {acc:.4f}, train_time = {t1 - t0:.2f}s")

for name, info in methods.items():
    X_clf = info["X_clf"]
    eval_logreg_on_embedding(X_clf, name)

plt.figure(figsize=(7, 4))
plt.bar(clf_acc.keys(), clf_acc.values())
plt.xticks(rotation=0)
plt.ylabel("Accuracy")
plt.title("Classification Accuracy on Different Embeddings")
plt.tight_layout()
plt.show()

# ==========================
# 9. RUNTIME SUMMARY (fit+transform)
# ==========================

print("\n=== Runtime summary (fit+transform for DR) ===")
for name, info in methods.items():
    print(f"{name:10s}: {info['runtime']:.2f}s")

plt.figure(figsize=(7, 4))
plt.bar(list(methods.keys()),
        [methods[k]["runtime"] for k in methods.keys()])
plt.xticks(rotation=0)
plt.ylabel("Time (s)")
plt.title("DR Runtime Comparison (fit+transform)")
plt.tight_layout()
plt.show()

# ==========================
# 10. RADAR CHART – TỔNG HỢP 5 TIÊU CHÍ
# ==========================

print("\n=== Radar chart – tổng hợp PCA / IPCA / t-SNE ===")

radar_methods = ["PCA", "IPCA", "t-SNE"]

# 5 tiêu chí:
# 1) Runtime (lower better)
# 2) Silhouette (higher better)
# 3) Stress (lower better)
# 4) Classification accuracy (higher better)
# 5) Interpretability (manual)

interpretability = {
    "PCA": 5,
    "IPCA": 4,
    "t-SNE": 1,
}

# Gather raw values
rt_vals   = np.array([methods[m]["runtime"] for m in radar_methods])
sil_vals  = np.array([sil_scores[m] for m in radar_methods])
str_vals  = np.array([stress_scores[m] for m in radar_methods])
acc_vals  = np.array([clf_acc[m] for m in radar_methods])
interp_vals = np.array([interpretability[m] for m in radar_methods], dtype=float)

# Normalize to [0,1]
eps = 1e-9
rt_norm   = (rt_vals.max() - rt_vals) / (rt_vals.max() - rt_vals.min() + eps)
str_norm  = (str_vals.max() - str_vals) / (str_vals.max() - str_vals.min() + eps)
sil_norm  = (sil_vals - sil_vals.min()) / (sil_vals.max() - sil_vals.min() + eps)
acc_norm  = (acc_vals - acc_vals.min()) / (acc_vals.max() - acc_vals.min() + eps)
interp_norm = (interp_vals - interp_vals.min()) / (interp_vals.max() - interp_vals.min() + eps)

metrics_matrix = np.vstack([rt_norm, sil_norm, str_norm, acc_norm, interp_norm]).T
metric_labels = ["Runtime", "Silhouette", "Stress", "Accuracy", "Interpretability"]

# Radar plot
num_metrics = len(metric_labels)
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
angles = np.concatenate([angles, [angles[0]]])  # close the loop

plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

for i, m in enumerate(radar_methods):
    vals = metrics_matrix[i]
    vals = np.concatenate([vals, [vals[0]]])
    ax.plot(angles, vals, label=m)
    ax.fill(angles, vals, alpha=0.1)

ax.set_thetagrids(angles[:-1] * 180 / np.pi, metric_labels)
ax.set_title("Radar Chart – PCA vs IPCA vs t-SNE (raw)", fontsize=14)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.show()

print("\n=== DONE: All extended outputs for PCA / IPCA / t-SNE (raw) generated ===")
