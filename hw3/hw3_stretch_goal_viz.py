"""
Stretch Goal Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
DATA_2016_2022 = Path("hw3_output/cleaned_data_2016_2022.csv")
DATA_2022_2026 = Path("hw3_output/cleaned_data_2022_2026.csv")
OUTPUT_DIR = Path("hw3_output/figures_stretch_goal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("="*70)
print("STRETCH GOAL: 10-Year Trend Comparison (2016-2026)")
print("="*70)

# ============================================================================
# Load Data
# ============================================================================
df_old = pd.read_csv(DATA_2016_2022)
df_new = pd.read_csv(DATA_2022_2026)

# Combine for 10-year view
df_all = pd.concat([df_old, df_new], ignore_index=True)
df_all = df_all[df_all["App_Type"].isin(["Social Services", "Construction/Development"])]

print(f"\n10-Year Data Overview:")
print(f"  2016-2022: {len(df_old)} records")
print(f"  2022-2026: {len(df_new)} records")
print(f"  Total: {len(df_all)} records")
print(f"  Year range: {df_all['Year'].min()}-{df_all['Year'].max()}")

# ============================================================================
# 10-Year Overview
# ============================================================================
print(f"\n{'='*70}")
print("Generating 10-Year Overview")
print(f"{'='*70}")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Application count trend (10 years)
ax1 = fig.add_subplot(gs[0, :])
years = sorted(df_all['Year'].unique())
counts = [len(df_all[df_all['Year'] == yr]) for yr in years]

# Color code by period
colors = ['steelblue' if yr < 2022 else 'coral' for yr in years]
bars = ax1.bar(range(len(years)), counts, color=colors, width=0.7)

ax1.set_xticks(range(len(years)))
ax1.set_xticklabels(years)
ax1.set_title('10-Year Application Trend (2016-2026)', 
             fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Number of Applications', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, count in enumerate(counts):
    ax1.text(i, count + 0.5, str(count), ha='center', fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', label='2016-2022'),
                   Patch(facecolor='coral', label='2022-2026')]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)

# Add period divider
ax1.axvline(x=5.5, color='red', linestyle='--', linewidth=2, alpha=0.5)

# 2. Applications by category (10 years)
ax2 = fig.add_subplot(gs[1, 0])
cat_yearly = df_all.groupby(['Year', 'App_Type']).size().unstack(fill_value=0)
cat_yearly.plot(kind='bar', ax=ax2, color=['steelblue', 'coral'], width=0.7)
ax2.set_title('10-Year Trend by Category', fontsize=13, fontweight='bold')
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Number of Applications', fontsize=11)
ax2.legend(title='Category')
ax2.set_xticklabels(cat_yearly.index, rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Average score trend (10 years)
ax3 = fig.add_subplot(gs[1, 1])
avg_scores = df_all.groupby('Year')['Total_Score'].mean()
ax3.plot(range(len(avg_scores)), avg_scores.values, 
        marker='o', linewidth=2.5, markersize=8, color='steelblue')
ax3.set_xticks(range(len(avg_scores)))
ax3.set_xticklabels(avg_scores.index)
ax3.set_title('10-Year Average Score Trend', fontsize=13, fontweight='bold')
ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Average Score', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.axvline(x=5.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

# Add value labels
for i, score in enumerate(avg_scores.values):
    ax3.text(i, score + 0.5, f'{score:.1f}', ha='center', fontsize=9)

# 4. Funding trends (10 years)
ax4 = fig.add_subplot(gs[2, 0])
funding = df_all.groupby('Year').agg({
    'Funding_Request': 'sum',
    'Funding_Award': 'sum'
}) / 1000000  # Convert to millions

x = np.arange(len(funding))
width = 0.35
ax4.bar(x - width/2, funding['Funding_Request'], width, 
       label='Requested', color='steelblue', alpha=0.8)
ax4.bar(x + width/2, funding['Funding_Award'], width, 
       label='Awarded', color='coral', alpha=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(funding.index, rotation=45)
ax4.set_title('10-Year Funding Trends (Millions)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Year', fontsize=11)
ax4.set_ylabel('Amount (Million $)', fontsize=11)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.axvline(x=5.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

# 5. Priority distribution (10 years)
ax5 = fig.add_subplot(gs[2, 1])
priority_yearly = df_all.groupby(['Year', 'Priority_Category']).size().unstack(fill_value=0)

main_priorities = ['ANGHP', 'EO', 'NI', 'HA']
priority_yearly_main = priority_yearly[[p for p in main_priorities if p in priority_yearly.columns]]

priority_yearly_main.plot(kind='area', stacked=True, ax=ax5, alpha=0.7,
                         color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
ax5.set_title('10-Year Priority Distribution', fontsize=13, fontweight='bold')
ax5.set_xlabel('Year', fontsize=11)
ax5.set_ylabel('Number of Applications', fontsize=11)
ax5.legend(title='Priority')
ax5.set_xticklabels(priority_yearly.index, rotation=45)
ax5.grid(True, alpha=0.3, axis='y')
ax5.axvline(x=5.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

plt.savefig(OUTPUT_DIR / 'stretch_10year_overview.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: stretch_10year_overview.png")

# ============================================================================
# Period Comparison (2016-2022 vs 2022-2026)
# ============================================================================
print(f"\n{'='*70}")
print("Generating Period Comparison")
print(f"{'='*70}")

df_old['Period'] = '2016-2022'
df_new['Period'] = '2022-2026'
df_comparison = pd.concat([df_old, df_new], ignore_index=True)

# Calculate comparison statistics
comparison_stats = df_comparison.groupby('Period').agg({
    'Organization': 'count',
    'Total_Score': 'mean',
    'Funding_Request': 'mean',
    'Funding_Award': 'mean'
}).round(2)

comparison_stats.columns = ['Total_Apps', 'Avg_Score', 'Avg_Request', 'Avg_Award']
print("\nPeriod Comparison:")
print(comparison_stats)

# By category
period_by_type = df_comparison.groupby(['Period', 'App_Type']).size().unstack(fill_value=0)
print("\nBy Category:")
print(period_by_type)

# By priority
period_by_priority = df_comparison.groupby(['Period', 'Priority_Category']).size().unstack(fill_value=0)
print("\nBy Priority:")
print(period_by_priority)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Total applications
periods = comparison_stats.index
x = np.arange(len(periods))
axes[0, 0].bar(x, comparison_stats['Total_Apps'], color=['steelblue', 'coral'], width=0.6)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(periods)
axes[0, 0].set_title('Total Applications Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Number of Applications')
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(comparison_stats['Total_Apps']):
    axes[0, 0].text(i, val + 2, f'{int(val)}', ha='center', fontweight='bold')

# 2. Average score
axes[0, 1].bar(x, comparison_stats['Avg_Score'], color=['steelblue', 'coral'], width=0.6)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(periods)
axes[0, 1].set_title('Average Score Comparison', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Average Score')
axes[0, 1].set_ylim([80, 90])
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(comparison_stats['Avg_Score']):
    axes[0, 1].text(i, val + 0.3, f'{val:.1f}', ha='center', fontweight='bold')

# 3. Average funding
width = 0.35
axes[0, 2].bar(x - width/2, comparison_stats['Avg_Request']/1000, width, 
              label='Request', color='steelblue')
axes[0, 2].bar(x + width/2, comparison_stats['Avg_Award']/1000, width, 
              label='Award', color='coral')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(periods)
axes[0, 2].set_title('Average Funding Comparison (K$)', fontsize=12, fontweight='bold')
axes[0, 2].set_ylabel('Amount (Thousand $)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3, axis='y')

# 4. By category
period_by_type.T.plot(kind='bar', ax=axes[1, 0], color=['steelblue', 'coral'])
axes[1, 0].set_title('Comparison by Category', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend(title='Period', fontsize=9)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 5. By priority
main_priorities_comp = [p for p in ['ANGHP', 'EO', 'NI', 'HA'] if p in period_by_priority.columns]
period_by_priority[main_priorities_comp].plot(kind='bar', ax=axes[1, 1],
                                              color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
axes[1, 1].set_title('Comparison by Priority', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Period')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend(title='Priority', fontsize=9)
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# 6. Top organizations
top_orgs_old = df_old.groupby('Organization').size().nlargest(10)
axes[1, 2].barh(range(len(top_orgs_old)), top_orgs_old.values, color='steelblue', alpha=0.7)
axes[1, 2].set_yticks(range(len(top_orgs_old)))
axes[1, 2].set_yticklabels([org[:25] + '...' if len(org) > 25 else org 
                           for org in top_orgs_old.index], fontsize=8)
axes[1, 2].set_title('Top 10 Orgs (2016-2022)', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Applications')
axes[1, 2].invert_yaxis()
axes[1, 2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'stretch_period_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: stretch_period_comparison.png")

# ============================================================================
# Summary Statistics
# ============================================================================
print(f"\n{'='*70}")
print("KEY FINDINGS")
print(f"{'='*70}")

# Calculate changes
pct_changes = {
    'Applications': (len(df_new) - len(df_old)) / len(df_old) * 100,
    'Avg_Score': df_new['Total_Score'].mean() - df_old['Total_Score'].mean(),
    'Avg_Request': (df_new['Funding_Request'].mean() - df_old['Funding_Request'].mean()) / df_old['Funding_Request'].mean() * 100,
    'Avg_Award': (df_new['Funding_Award'].mean() - df_old['Funding_Award'].mean()) / df_old['Funding_Award'].mean() * 100
}

print(f"\nChanges from 2016-2022 to 2022-2026:")
print(f"  • Application volume: {pct_changes['Applications']:+.1f}%")
print(f"  • Average score: {pct_changes['Avg_Score']:+.2f} points")
print(f"  • Average request: {pct_changes['Avg_Request']:+.1f}%")
print(f"  • Average award: {pct_changes['Avg_Award']:+.1f}%")

print(f"\n{'='*70}")
print("VISUALIZATION COMPLETE")
print(f"{'='*70}")
print(f"\nGenerated files:")
print(f"  • stretch_10year_overview.png")
print(f"  • stretch_period_comparison.png")
print("="*70)
