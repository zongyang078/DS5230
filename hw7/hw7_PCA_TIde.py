# Zongyang Li
# 03/20/2026

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

os.makedirs("../hw7_output", exist_ok=True)


# Part 1: PCA Analysis

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
        print(f"  {i + 1}. {name}")

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
        print(f"  PC{i + 1}: {var:.4f} ({var * 100:.2f}%)  |  Cumulative: {cumulative:.4f} ({cumulative * 100:.2f}%)")

    # 6. Eigenvectors (components) for each PC
    print(f"\n{'─' * 50}")
    print("Eigenvectors (Loadings) for each PC:")
    print(f"{'─' * 50}")

    loadings_df = pd.DataFrame(
        pc.components_.T,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(n_components)]
    )
    print(loadings_df.round(4).to_string())

    # 7. Which variables load into which PC
    print(f"\n{'─' * 50}")
    print("Dominant Variables per PC (|loading| > 0.3):")
    print(f"{'─' * 50}")
    for i in range(n_components):
        pc_loadings = pc.components_[i]
        sorted_indices = np.argsort(np.abs(pc_loadings))[::-1]
        dominant = [(feature_names[j], pc_loadings[j]) for j in sorted_indices if abs(pc_loadings[j]) > 0.3]
        print(f"\n  PC{i + 1} (explains {pc.explained_variance_ratio_[i] * 100:.2f}% variance):")
        if dominant:
            for name, val in dominant:
                direction = "+" if val > 0 else "-"
                print(f"    {direction} {name}: {val:.4f}")
        else:
            print(f"    No variable with |loading| > 0.3")

    # Plots

    # Plot 1: Scree Plot + Biplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pcs = [f"PC{i + 1}" for i in range(n_components)]
    variances = pc.explained_variance_ratio_
    cumulative_var = np.cumsum(variances)

    axes[0].bar(pcs, variances, color='steelblue', edgecolor='black')
    axes[0].plot(pcs, cumulative_var, 'ro-', linewidth=2)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained")
    axes[0].set_title(f"{dataset_name} - Scree Plot")
    axes[0].axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='10% threshold')
    axes[0].legend(['Individual', 'Cumulative', '10% threshold'])
    axes[0].tick_params(axis='x', rotation=45)

    # Biplot (PC1 vs PC2)
    scores = pca.transform(X)
    axes[1].scatter(scores[:, 0], scores[:, 1], alpha=0.3, s=10, color='steelblue')

    scale = max(abs(scores[:, 0]).max(), abs(scores[:, 1]).max()) * 0.8
    for j, name in enumerate(feature_names):
        axes[1].arrow(0, 0,
                      pc.components_[0, j] * scale,
                      pc.components_[1, j] * scale,
                      head_width=scale * 0.03, head_length=scale * 0.02,
                      fc='red', ec='red', alpha=0.7)
        axes[1].text(pc.components_[0, j] * scale * 1.12,
                     pc.components_[1, j] * scale * 1.12,
                     name, fontsize=7, color='red', ha='center')

    axes[1].set_xlabel(f"PC1 ({variances[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({variances[1] * 100:.1f}%)")
    axes[1].set_title(f"{dataset_name} - Biplot (PC1 vs PC2)")
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].axvline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    filename = dataset_name.lower().replace(" ", "_")
    plt.savefig(f"../hw7_output/{filename}_pca.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlots saved: {filename}_pca.png")

    # Plot 2: Heatmap of loadings
    fig, ax = plt.subplots(figsize=(10, max(6, n_components * 0.6)))
    im = ax.imshow(np.abs(loadings_df.values), cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_components))
    ax.set_xticklabels([f"PC{i + 1}" for i in range(n_components)], rotation=45)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)

    for i in range(len(feature_names)):
        for j in range(n_components):
            val = loadings_df.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7,
                    color='white' if abs(val) > 0.4 else 'black')

    plt.colorbar(im, label='|Loading|')
    ax.set_title(f"{dataset_name} - PCA Loadings Heatmap")
    plt.tight_layout()
    plt.savefig(f"../hw7_output/{filename}_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {filename}_heatmap.png")

    return pca, loadings_df


# Run PCA on Concrete Dataset
print("\n\n")
pca_concrete, loadings_concrete = run_pca_analysis(
    "../data/concrete.csv",
    "Concrete",
    drop_cols=None
)

# Run PCA on Abalone Dataset
print("\n\n")
pca_abalone, loadings_abalone = run_pca_analysis(
    "../data/abalone.csv",
    "Abalone",
    drop_cols=["Type"]
)


# Part 2: Tide Prediction using Fourier Transform

print("\n\n")
print("=" * 70)
print("Tide Prediction: Fourier Transform Analysis")
print("=" * 70)

# 1. Load tide data
tide_df = pd.read_csv("../data/CO-OPS_8418150_wl.csv")
print(f"\nTide dataset shape: {tide_df.shape}")
print(f"Columns: {list(tide_df.columns)}")
print(f"\nFirst 5 rows:\n{tide_df.head()}")

# Extract the Verified column
verified = pd.to_numeric(tide_df["Verified (ft)"], errors='coerce')
print(f"\nVerified data: {verified.count()} valid values out of {len(verified)}")

# Fill any NaN values with interpolation
verified = verified.interpolate().values

# Create time axis (6-minute intervals)
n = len(verified)
dt = 6  # minutes per sample
time_hours = np.arange(n) * dt / 60  # time in hours

# 2. Plot original verified data
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(time_hours, verified, color='green', linewidth=0.5, alpha=0.8)
ax.set_xlabel("Time (hours from Jan 1, 2024)")
ax.set_ylabel("Water Level (ft, MLLW)")
ax.set_title("Portland, ME - Verified Tidal Data (January 2024)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../hw7_output/tide_original.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: tide_original.png")

# 3. Run FFT on Verified data
fft_vals = np.fft.fft(verified)
fft_magnitudes = np.abs(fft_vals) / n  # normalized magnitudes
freqs = np.fft.fftfreq(n, d=dt)  # frequencies in cycles per minute

# Only look at positive frequencies
pos_mask = freqs > 0
pos_freqs = freqs[pos_mask]
pos_magnitudes = fft_magnitudes[pos_mask] * 2  # multiply by 2 for one-sided spectrum

# Convert frequency to period in hours
pos_periods_hours = 1 / (pos_freqs * 60)

print(f"\n{'─' * 50}")
print("Top 10 Frequency Peaks:")
print(f"{'─' * 50}")
top_indices = np.argsort(pos_magnitudes)[::-1][:10]
for i, idx in enumerate(top_indices):
    period_h = pos_periods_hours[idx]
    freq_cpm = pos_freqs[idx]
    mag = pos_magnitudes[idx]
    print(f"  Peak {i + 1}: Period = {period_h:.2f} hours ({period_h / 24:.2f} days), "
          f"Magnitude = {mag:.4f}, Freq = {freq_cpm:.8f} cycles/min")

# 4. Stem plot of FFT magnitudes
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Full spectrum stem plot
markerline, stemlines, baseline = axes[0].stem(pos_freqs * 60, pos_magnitudes,
                                               linefmt='b-', markerfmt='b.', basefmt='k-')
plt.setp(stemlines, linewidth=0.5)
plt.setp(markerline, markersize=2)
axes[0].set_xlabel("Frequency (cycles per hour)")
axes[0].set_ylabel("Magnitude")
axes[0].set_title("FFT of Verified Tidal Data - Full Spectrum")
axes[0].grid(True, alpha=0.3)

# Zoomed stem plot (dominant frequencies only)
zoom_mask = pos_freqs * 60 < 0.5  # periods > 2 hours
markerline, stemlines, baseline = axes[1].stem(
    pos_freqs[zoom_mask] * 60, pos_magnitudes[zoom_mask],
    linefmt='b-', markerfmt='bo', basefmt='k-'
)
plt.setp(stemlines, linewidth=0.8)
plt.setp(markerline, markersize=3)
axes[1].set_xlabel("Frequency (cycles per hour)")
axes[1].set_ylabel("Magnitude")
axes[1].set_title("FFT of Verified Tidal Data - Zoomed (Dominant Frequencies)")
axes[1].grid(True, alpha=0.3)

# Annotate major peaks
for idx in top_indices[:5]:
    if pos_freqs[idx] * 60 < 0.5:
        period_h = pos_periods_hours[idx]
        axes[1].annotate(
            f"{period_h:.1f}h",
            xy=(pos_freqs[idx] * 60, pos_magnitudes[idx]),
            xytext=(pos_freqs[idx] * 60 + 0.01, pos_magnitudes[idx] + 0.05),
            fontsize=9, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1)
        )

plt.tight_layout()
plt.savefig("../hw7_output/tide_fft_stem.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: tide_fft_stem.png")

# 5. Apply threshold and inverse FFT for tide prediction
# Threshold selection rationale:
#   The magnitude distribution shows that 99.5% of frequencies have magnitude
#   below ~0.11. A threshold of 0.15 sits just above this 99.5th percentile,
#   ensuring we keep only the physically meaningful tidal harmonics while
#   filtering out noise. This keeps just 29 frequencies (0.4% of total) but
#   achieves RMSE = 0.36 ft, which is ~2.5% of the tidal range (0-14 ft).

threshold = 0.15
print(f"\n{'─' * 50}")
print(f"Threshold Analysis (threshold = {threshold})")
print(f"{'─' * 50}")

# Show magnitude distribution to justify threshold
print(f"\n  Magnitude distribution (positive frequencies):")
for p in [50, 75, 90, 95, 99, 99.5]:
    print(f"    {p}th percentile: {np.percentile(pos_magnitudes, p):.4f}")
print(f"    Max: {pos_magnitudes.max():.4f}")
print(f"  -> Threshold {threshold} is above 99.5th percentile, keeping only dominant harmonics")

# Apply threshold: zero out small frequencies
fft_filtered = fft_vals.copy()
magnitude_normalized = np.abs(fft_vals) / n
mask = magnitude_normalized >= threshold / 2  # divide by 2 because we doubled for one-sided
mask[0] = True  # always keep DC (mean)
fft_filtered[~mask] = 0

n_kept = np.sum(mask)
n_total = len(fft_vals)
print(f"\n  Frequencies kept: {n_kept} out of {n_total} ({n_kept / n_total * 100:.1f}%)")

# List the kept frequencies
kept_indices = np.where(mask & (freqs > 0))[0]
print(f"\n  Significant frequencies kept:")
for idx in kept_indices:
    period_h = 1 / (freqs[idx] * 60)
    mag = np.abs(fft_vals[idx]) / n * 2
    print(f"    Period = {period_h:.2f} hours ({period_h / 24:.2f} days), Magnitude = {mag:.4f}")

# Run inverse FFT
tide_predicted = np.fft.ifft(fft_filtered).real

print(f"\n  Original data range: [{verified.min():.2f}, {verified.max():.2f}]")
print(f"  Predicted data range: [{tide_predicted.min():.2f}, {tide_predicted.max():.2f}]")
print(f"  RMSE: {np.sqrt(np.mean((verified - tide_predicted) ** 2)):.4f} ft")

# 6. Plot comparison: Original vs Predicted
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Full month comparison
axes[0].plot(time_hours, verified, color='green', linewidth=0.5, alpha=0.7, label='Verified (Actual)')
axes[0].plot(time_hours, tide_predicted, color='red', linewidth=0.8, alpha=0.8, label='FFT Prediction')
axes[0].set_xlabel("Time (hours from Jan 1, 2024)")
axes[0].set_ylabel("Water Level (ft, MLLW)")
axes[0].set_title("Tide Prediction vs Verified Data - Full Month")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Zoomed view (first 7 days)
zoom_hours = 168
zoom_idx = time_hours <= zoom_hours
axes[1].plot(time_hours[zoom_idx], verified[zoom_idx], color='green', linewidth=1, alpha=0.7, label='Verified (Actual)')
axes[1].plot(time_hours[zoom_idx], tide_predicted[zoom_idx], color='red', linewidth=1.2, alpha=0.8,
             label='FFT Prediction')
axes[1].set_xlabel("Time (hours from Jan 1, 2024)")
axes[1].set_ylabel("Water Level (ft, MLLW)")
axes[1].set_title("Tide Prediction vs Verified Data - First 7 Days (Zoomed)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../hw7_output/tide_prediction.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: tide_prediction.png")

# 7. Plot the filtered FFT stem (after thresholding)
fig, ax = plt.subplots(figsize=(14, 4))
filtered_magnitudes = np.abs(fft_filtered[pos_mask]) / n * 2
zoom_mask2 = pos_freqs * 60 < 0.5
markerline, stemlines, baseline = ax.stem(
    pos_freqs[zoom_mask2] * 60, filtered_magnitudes[zoom_mask2],
    linefmt='b-', markerfmt='bo', basefmt='k-'
)
plt.setp(stemlines, linewidth=0.8)
plt.setp(markerline, markersize=3)
ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold = {threshold}')
ax.set_xlabel("Frequency (cycles per hour)")
ax.set_ylabel("Magnitude")
ax.set_title(f"FFT After Thresholding (threshold = {threshold}, {n_kept} frequencies kept)")
ax.legend()
ax.grid(True, alpha=0.3)

# Annotate only top 5 kept peaks to avoid clutter
top_kept = sorted(kept_indices, key=lambda i: np.abs(fft_vals[i]), reverse=True)[:5]
label_positions_used = []
for idx in top_kept:
    freq_cph = freqs[idx] * 60
    if freq_cph < 0.5:
        period_h = 1 / (freqs[idx] * 60)
        mag = np.abs(fft_vals[idx]) / n * 2
        # Smart offset: alternate left/right and adjust y to avoid overlap
        tx = freq_cph + 0.012
        ty = mag + 0.15
        # Check if too close to existing label
        for px, py in label_positions_used:
            if abs(tx - px) < 0.015 and abs(ty - py) < 0.3:
                ty = py + 0.3
        label_positions_used.append((tx, ty))
        ax.annotate(
            f"{period_h:.1f}h",
            xy=(freq_cph, mag),
            xytext=(tx, ty),
            fontsize=9, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1)
        )

plt.tight_layout()
plt.savefig("../hw7_output/tide_fft_filtered.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved: tide_fft_filtered.png")

print("\n" + "=" * 70)
print("All analysis complete! Check hw7_output/ for all plots.")
print("=" * 70)