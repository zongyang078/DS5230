import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ============================================================
# Helper Functions
# ============================================================

def run_pca_analysis(filepath, dataset_name, drop_cols=None):
    """Run full PCA analysis on a dataset and generate all required outputs."""
    
    print("=" * 70)
    print(f"PCA Analysis: {dataset_name}")
    print("=" * 70)
    
    # 1. Load data
    df = pd.read_csv(filepath)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    
    # 2. Prepare data - drop non-numeric or target columns
    if drop_cols:
        df_numeric = df.drop(columns=drop_cols)
    else:
        df_numeric = df.copy()
    
    feature_names = list(df_numeric.columns)
    data = df_numeric.values
    n_components = len(feature_names)
    
    print(f"\nFeatures used for PCA ({n_components} total):")
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name}")
    
    # 3. Standardize data
    X = StandardScaler().fit_transform(data)
    
    # 4. Run PCA
    pca = PCA(n_components=n_components)
    pc = pca.fit(X)
    
    # 5. Explained variance ratio for each PC
    print(f"\n{'─' * 50}")
    print("Explained Variance Ratio per PC:")
    print(f"{'─' * 50}")
    cumulative = 0
    for i, var in enumerate(pc.explained_variance_ratio_):
        cumulative += var
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)  |  Cumulative: {cumulative:.4f} ({cumulative*100:.2f}%)")
    
    # 6. Eigenvectors (components) for each PC
    print(f"\n{'─' * 50}")
    print("Eigenvectors (Loadings) for each PC:")
    print(f"{'─' * 50}")
    
    # Create a nice DataFrame for display
    loadings_df = pd.DataFrame(
        pc.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    print(loadings_df.round(4).to_string())
    
    # 7. Which variables load into which PC
    print(f"\n{'─' * 50}")
    print("Dominant Variables per PC (|loading| > 0.3):")
    print(f"{'─' * 50}")
    for i in range(n_components):
        pc_loadings = pc.components_[i]
        # Sort by absolute value, descending
        sorted_indices = np.argsort(np.abs(pc_loadings))[::-1]
        dominant = [(feature_names[j], pc_loadings[j]) for j in sorted_indices if abs(pc_loadings[j]) > 0.3]
        print(f"\n  PC{i+1} (explains {pc.explained_variance_ratio_[i]*100:.2f}% variance):")
        if dominant:
            for name, val in dominant:
                direction = "+" if val > 0 else "-"
                print(f"    {direction} {name}: {val:.4f}")
        else:
            print(f"    No variable with |loading| > 0.3")
    
    # ============================================================
    # Plots
    # ============================================================
    
    # Plot 1: Scree Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    pcs = [f"PC{i+1}" for i in range(n_components)]
    variances = pc.explained_variance_ratio_
    cumulative_var = np.cumsum(variances)
    
    axes[0].bar(pcs, variances, color='steelblue', edgecolor='black')
    axes[0].plot(pcs, cumulative_var, 'ro-', linewidth=2)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained")
    axes[0].set_title(f"{dataset_name} - Scree Plot")
    axes[0].axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='10% threshold')
    axes[0].legend(['Cumulative', 'Individual', '10% threshold'])
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Biplot (PC1 vs PC2)
    scores = pca.transform(X)
    axes[1].scatter(scores[:, 0], scores[:, 1], alpha=0.3, s=10, color='steelblue')
    
    # Overlay loading vectors
    scale = max(abs(scores[:, 0]).max(), abs(scores[:, 1]).max()) * 0.8
    for j, name in enumerate(feature_names):
        axes[1].arrow(0, 0,
                       pc.components_[0, j] * scale,
                       pc.components_[1, j] * scale,
                       head_width=scale*0.03, head_length=scale*0.02,
                       fc='red', ec='red', alpha=0.7)
        axes[1].text(pc.components_[0, j] * scale * 1.12,
                      pc.components_[1, j] * scale * 1.12,
                      name, fontsize=7, color='red', ha='center')
    
    axes[1].set_xlabel(f"PC1 ({variances[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({variances[1]*100:.1f}%)")
    axes[1].set_title(f"{dataset_name} - Biplot (PC1 vs PC2)")
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].axvline(0, color='gray', linewidth=0.5)
    
    plt.tight_layout()
    filename = dataset_name.lower().replace(" ", "_")
    plt.savefig(f"/home/claude/{filename}_pca.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlots saved: {filename}_pca.png")
    
    # Plot 3: Heatmap of loadings
    fig, ax = plt.subplots(figsize=(10, max(6, n_components * 0.6)))
    im = ax.imshow(np.abs(loadings_df.values), cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_components))
    ax.set_xticklabels([f"PC{i+1}" for i in range(n_components)], rotation=45)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    
    # Add text annotations
    for i in range(len(feature_names)):
        for j in range(n_components):
            val = loadings_df.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7,
                    color='white' if abs(val) > 0.4 else 'black')
    
    plt.colorbar(im, label='|Loading|')
    ax.set_title(f"{dataset_name} - PCA Loadings Heatmap")
    plt.tight_layout()
    plt.savefig(f"/home/claude/{filename}_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {filename}_heatmap.png")
    
    return pca, loadings_df


# ============================================================
# Run PCA on Concrete Dataset
# ============================================================
print("\n\n")
pca_concrete, loadings_concrete = run_pca_analysis(
    "/mnt/user-data/uploads/concrete.csv",
    "Concrete",
    drop_cols=None  # All columns are numeric features (including CompressiveStrength)
)

# ============================================================
# Run PCA on Abalone Dataset
# ============================================================
print("\n\n")
pca_abalone, loadings_abalone = run_pca_analysis(
    "/mnt/user-data/uploads/abalone.csv",
    "Abalone",
    drop_cols=["Type"]  # Drop the categorical column
)
