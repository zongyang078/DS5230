import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# 1 Load data
csv_path = "data/sleep_study_60.csv"
df = pd.read_csv(csv_path)

# 2 Column name helpers (robust to long Google-Forms-style headers)
def find_col_contains(df, keyword):
    matches = [c for c in df.columns if keyword.lower() in c.lower()]
    if not matches:
        raise ValueError(f"Could not find any column containing keyword: {keyword}")
    return matches[0]

col_sleep = find_col_contains(df, "sleep per night")
col_study = find_col_contains(df, "coursework per day")
col_regular = find_col_contains(df, "sleep schedule to be regular")
col_chrono = find_col_contains(df, "sleep preference")
col_eff = find_col_contains(df, "academic efficiency")
col_stress = find_col_contains(df, "stressed did you feel")

# 3 Clean / parse Likert labels -> numeric (1..5)
def likert_label_to_int(x):
    """
    Convert strings like '3 – Moderate' or '3 - Moderate' to integer 3.
    If already numeric, keep it.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return int(x)
    s = str(x).strip()
    m = re.match(r"^\s*([1-5])\s*[–-]", s)  # '–' or '-' dash
    if m:
        return int(m.group(1))
    # fallback: try to extract first digit 1..5
    m2 = re.search(r"([1-5])", s)
    if m2:
        return int(m2.group(1))
    return np.nan

df["sleep_hours"] = pd.to_numeric(df[col_sleep], errors="coerce")
df["study_hours"] = pd.to_numeric(df[col_study], errors="coerce")
df["regular_sleep"] = df[col_regular].astype(str).str.strip()
df["chronotype"] = df[col_chrono].astype(str).str.strip()
df["efficiency"] = df[col_eff].apply(likert_label_to_int)
df["stress"] = df[col_stress].apply(likert_label_to_int)

# Drop rows with missing essentials
df_clean = df.dropna(subset=["sleep_hours", "study_hours", "efficiency", "stress", "regular_sleep"]).copy()

# 4 Plot helpers
def savefig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=200)
    plt.close()

# 5 Histograms
plt.figure()
plt.hist(df_clean["sleep_hours"], bins=np.arange(4.5, 9.6, 0.5))
plt.title("Distribution of Sleep Duration (Hours per Night)")
plt.xlabel("Sleep hours")
plt.ylabel("Count")
savefig("fig1_hist_sleep_hours.png")

plt.figure()
plt.hist(df_clean["study_hours"], bins=np.arange(1.0, 7.1, 0.5))
plt.title("Distribution of Study Time (Hours per Day)")
plt.xlabel("Study hours")
plt.ylabel("Count")
savefig("fig2_hist_study_hours.png")

# 6 Box plot: stress by sleep regularity
# Ensure consistent order
groups = ["Yes", "No"]
data = [df_clean.loc[df_clean["regular_sleep"] == g, "stress"] for g in groups]

plt.figure()
plt.boxplot(data, labels=groups)
plt.title("Stress Level by Sleep Schedule Regularity")
plt.xlabel("Regular sleep schedule?")
plt.ylabel("Stress (1–5)")
savefig("fig3_box_stress_by_regular.png")

# Optional: efficiency by regularity
data_eff = [df_clean.loc[df_clean["regular_sleep"] == g, "efficiency"] for g in groups]
plt.figure()
plt.boxplot(data_eff, labels=groups)
plt.title("Academic Efficiency by Sleep Schedule Regularity")
plt.xlabel("Regular sleep schedule?")
plt.ylabel("Efficiency (1–5)")
savefig("fig3b_box_eff_by_regular.png")

# 7 Scatter + regression line
def scatter_with_regression(x, y, x_label, y_label, title, out_name):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Fit y = a + b*x
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    b0, b1 = model.params[0], model.params[1]

    # Regression line across range
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = b0 + b1 * x_line
    plt.plot(x_line, y_line)

    savefig(out_name)
    return model

m1 = scatter_with_regression(
    df_clean["sleep_hours"], df_clean["efficiency"],
    "Sleep hours (per night)", "Efficiency (1–5)",
    "Sleep Duration vs Academic Efficiency (with Regression Line)",
    "fig4_scatter_sleep_vs_efficiency.png"
)

m2 = scatter_with_regression(
    df_clean["sleep_hours"], df_clean["stress"],
    "Sleep hours (per night)", "Stress (1–5)",
    "Sleep Duration vs Stress Level (with Regression Line)",
    "fig5_scatter_sleep_vs_stress.png"
)

# Optional: study vs stress
m3 = scatter_with_regression(
    df_clean["study_hours"], df_clean["stress"],
    "Study hours (per day)", "Stress (1–5)",
    "Study Time vs Stress Level (with Regression Line)",
    "fig5b_scatter_study_vs_stress.png"
)

# 8 Correlation (Pearson)
corr_sleep_eff = df_clean["sleep_hours"].corr(df_clean["efficiency"])
corr_sleep_stress = df_clean["sleep_hours"].corr(df_clean["stress"])
corr_study_eff = df_clean["study_hours"].corr(df_clean["efficiency"])
corr_study_stress = df_clean["study_hours"].corr(df_clean["stress"])

print("Pearson correlations:")
print(f"  corr(sleep, efficiency) = {corr_sleep_eff:.3f}")
print(f"  corr(sleep, stress)     = {corr_sleep_stress:.3f}")
print(f"  corr(study, efficiency) = {corr_study_eff:.3f}")
print(f"  corr(study, stress)     = {corr_study_stress:.3f}")

# 9) Multiple regression
# Efficiency ~ sleep + study
# Stress ~ sleep + study + regular_sleep

# Efficiency model
X_eff = df_clean[["sleep_hours", "study_hours"]].copy()
X_eff = sm.add_constant(X_eff)
y_eff = df_clean["efficiency"]
model_eff = sm.OLS(y_eff, X_eff).fit()

print("\nMultiple regression: Efficiency ~ Sleep + Study")
print(model_eff.summary())

# Stress model with regular_sleep as 0/1 (Yes=1)
df_clean["regular_yes"] = (df_clean["regular_sleep"].str.lower() == "yes").astype(int)
X_stress = df_clean[["sleep_hours", "study_hours", "regular_yes"]].copy()
X_stress = sm.add_constant(X_stress)
y_stress = df_clean["stress"]
model_stress = sm.OLS(y_stress, X_stress).fit()

print("\nMultiple regression: Stress ~ Sleep + Study + Regular(Yes=1)")
print(model_stress.summary())

print("\nDone. Saved figures:")
print("  fig1_hist_sleep_hours.png")
print("  fig2_hist_study_hours.png")
print("  fig3_box_stress_by_regular.png")
print("  fig3b_box_eff_by_regular.png")
print("  fig4_scatter_sleep_vs_efficiency.png")
print("  fig5_scatter_sleep_vs_stress.png")
print("  fig5b_scatter_study_vs_stress.png")