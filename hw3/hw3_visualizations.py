"""
Generates all required visualizations per Rowen's requirements

Requirements from Rowen:
1. Number of applications compared per year
2. Spread of scores per year
3. Number of applications submitted (and/or funded) per organization
4. Spread of funding requests, including breakdown of scoring sections
5. Number and type of priority categories

Note: Separate findings by application category (Social Services vs Construction/Development)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
DATA_FILE = Path("hw3_output/cleaned_data_2022_2026.csv")
OUTPUT_DIR = Path("hw3_output/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

print("="*70)
print("HW3 VISUALIZATION - CDBG Applications Analysis")
print("="*70)

# Load cleaned data
df = pd.read_csv(DATA_FILE)
print(f"\nLoaded {len(df)} records")
print(f"Years: {df['Year'].min()}-{df['Year'].max()}")
print(f"Categories: {df['App_Type'].unique()}")

# ============================================================================
# Requirement 1: Number of Applications Compared Per Year
# ============================================================================
print(f"\n{'='*70}")
print("1. NUMBER OF APPLICATIONS PER YEAR")
print(f"{'='*70}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1a. Total applications per year
apps_by_year = df.groupby('Year').size()
axes[0].plot(apps_by_year.index, apps_by_year.values, 
             marker='o', linewidth=2.5, markersize=10, color='steelblue')
axes[0].set_title('Total Applications Per Year', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Number of Applications')
axes[0].grid(True, alpha=0.3)

# Add value labels
for year, count in zip(apps_by_year.index, apps_by_year.values):
    axes[0].text(year, count + 0.5, str(count), ha='center', fontweight='bold')

# 1b. Applications by category per year
apps_by_cat = df.groupby(['Year', 'App_Type']).size().unstack(fill_value=0)
apps_by_cat.plot(kind='bar', ax=axes[1], width=0.7)
axes[1].set_title('Applications Per Year by Category', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of Applications')
axes[1].legend(title='Category')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '1_applications_per_year.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 1_applications_per_year.png")

# ============================================================================
# Requirement 2: Spread of Scores Per Year
# ============================================================================
print(f"\n{'='*70}")
print("2. SPREAD OF SCORES PER YEAR")
print(f"{'='*70}")

# Filter data with scores
df_scores = df[df['Total_Score'].notna()].copy()
print(f"Records with scores: {len(df_scores)}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2a. Box plot of scores by year for Social Services
ss_data = df_scores[df_scores['App_Type'] == 'Social Services']
if len(ss_data) > 0:
    ss_scores_by_year = [ss_data[ss_data['Year'] == yr]['Total_Score'].values 
                         for yr in sorted(ss_data['Year'].unique())]
    bp1 = axes[0, 0].boxplot(ss_scores_by_year, 
                              labels=sorted(ss_data['Year'].unique()),
                              patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    axes[0, 0].set_title('Score Distribution - Social Services', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Total Score')
    axes[0, 0].grid(True, alpha=0.3)

# 2b. Box plot of scores by year for Construction/Development
con_data = df_scores[df_scores['App_Type'] == 'Construction/Development']
if len(con_data) > 0:
    con_scores_by_year = [con_data[con_data['Year'] == yr]['Total_Score'].values 
                          for yr in sorted(con_data['Year'].unique())]
    bp2 = axes[0, 1].boxplot(con_scores_by_year, 
                              labels=sorted(con_data['Year'].unique()),
                              patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    axes[0, 1].set_title('Score Distribution - Construction/Development', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Total Score')
    axes[0, 1].grid(True, alpha=0.3)

# 2c. Average score trends by category
avg_scores = df_scores.groupby(['Year', 'App_Type'])['Total_Score'].mean().unstack()
avg_scores.plot(ax=axes[1, 0], marker='o', linewidth=2, markersize=8)
axes[1, 0].set_title('Average Score Trends by Category', 
                    fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Average Score')
axes[1, 0].legend(title='Category')
axes[1, 0].grid(True, alpha=0.3)

# 2d. Score distribution histogram
axes[1, 1].hist([ss_data['Total_Score'], con_data['Total_Score']], 
               bins=15, alpha=0.7, 
               label=['Social Services', 'Construction/Development'])
axes[1, 1].set_title('Overall Score Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Total Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '2_score_spread.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 2_score_spread.png")

# ============================================================================
# Requirement 3: Applications Per Organization (Submitted and Funded)
# ============================================================================
print(f"\n{'='*70}")
print("3. APPLICATIONS PER ORGANIZATION")
print(f"{'='*70}")

# Calculate statistics per organization
org_stats = df.groupby('Organization').agg({
    'Year': 'count',  # Total applications
    'Funding_Award': lambda x: x.notna().sum()  # Funded applications
}).rename(columns={'Year': 'Total_Apps', 'Funding_Award': 'Funded_Apps'})

org_stats = org_stats.sort_values('Total_Apps', ascending=False).head(15)
org_stats['Funding_Rate'] = (org_stats['Funded_Apps'] / org_stats['Total_Apps'] * 100).round(1)

print(f"\nTop 15 organizations:")
print(org_stats)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 3a. Top organizations by total applications
axes[0].barh(range(len(org_stats)), org_stats['Total_Apps'], color='steelblue')
axes[0].set_yticks(range(len(org_stats)))
axes[0].set_yticklabels([org[:40] + '...' if len(org) > 40 else org 
                        for org in org_stats.index], fontsize=9)
axes[0].set_xlabel('Total Applications')
axes[0].set_title('Top 15 Organizations by Application Count', 
                  fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# 3b. Applications submitted vs funded
x = np.arange(len(org_stats.head(10)))
width = 0.35
axes[1].barh(x, org_stats.head(10)['Total_Apps'], width, 
            label='Submitted', color='steelblue')
axes[1].barh(x + width, org_stats.head(10)['Funded_Apps'], width, 
            label='Funded', color='orange')
axes[1].set_yticks(x + width / 2)
axes[1].set_yticklabels([org[:35] + '...' if len(org) > 35 else org 
                        for org in org_stats.head(10).index], fontsize=9)
axes[1].set_xlabel('Number of Applications')
axes[1].set_title('Top 10 Organizations: Submitted vs Funded', 
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '3_organizations.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 3_organizations.png")

# ============================================================================
# Requirement 4: Spread of Funding Requests + Scoring Breakdown
# ============================================================================
print(f"\n{'='*70}")
print("4. FUNDING REQUESTS & SCORING BREAKDOWN")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4a. Funding request distribution by year (box plot)
ss_funding = [df[df['Year'] == yr]['Funding_Request'].dropna().values 
              for yr in sorted(df['Year'].unique())]
bp = axes[0, 0].boxplot(ss_funding, labels=sorted(df['Year'].unique()),
                        patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
axes[0, 0].set_title('Funding Request Distribution by Year', 
                    fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Funding Request ($)')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 4b. Average funding trends
avg_funding = df.groupby(['Year', 'App_Type']).agg({
    'Funding_Request': 'mean',
    'Funding_Award': 'mean'
}).reset_index()

for app_type in ['Social Services', 'Construction/Development']:
    subset = avg_funding[avg_funding['App_Type'] == app_type]
    axes[0, 1].plot(subset['Year'], subset['Funding_Request'], 
                   marker='o', label=f'{app_type} - Request', linewidth=2)
    axes[0, 1].plot(subset['Year'], subset['Funding_Award'], 
                   marker='s', linestyle='--', label=f'{app_type} - Award', linewidth=2)

axes[0, 1].set_title('Average Funding Trends', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Average Amount ($)')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 4c. SCORING BREAKDOWN - Average scores by category
# This addresses "breakdown of scoring sections" requirement
score_cols = ['Score_Impact', 'Score_Principles', 'Score_Capacity', 'Score_Collab']
df_with_breakdown = df[df['Score_Impact'].notna()].copy()

if len(df_with_breakdown) > 0:
    score_breakdown = df_with_breakdown.groupby('App_Type')[score_cols].mean()
    
    x = np.arange(len(score_cols))
    width = 0.35
    
    if 'Social Services' in score_breakdown.index:
        axes[1, 0].bar(x - width/2, score_breakdown.loc['Social Services'], 
                      width, label='Social Services', color='steelblue')
    if 'Construction/Development' in score_breakdown.index:
        axes[1, 0].bar(x + width/2, score_breakdown.loc['Construction/Development'], 
                      width, label='Construction/Development', color='coral')
    
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['Impact\n(30pts)', 'Principles\n(30pts)', 
                                'Capacity\n(25pts)', 'Collab\n(15pts)'])
    axes[1, 0].set_title('Scoring Breakdown by Category', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Average Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    print(f"\nScoring breakdown (average):")
    print(score_breakdown.round(2))
else:
    axes[1, 0].text(0.5, 0.5, 'Scoring breakdown data\nnot available', 
                   ha='center', va='center', fontsize=12)
    axes[1, 0].set_title('Scoring Breakdown by Category', 
                        fontsize=12, fontweight='bold')

# 4d. Total funding by category
total_funding = df.groupby('App_Type').agg({
    'Funding_Request': 'sum',
    'Funding_Award': 'sum'
}) / 1000000  # Convert to millions

x = np.arange(len(total_funding))
width = 0.35
axes[1, 1].bar(x - width/2, total_funding['Funding_Request'], width, 
              label='Requested', color='steelblue')
axes[1, 1].bar(x + width/2, total_funding['Funding_Award'], width, 
              label='Awarded', color='orange')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(total_funding.index, rotation=15, ha='right')
axes[1, 1].set_title('Total Funding by Category', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Amount (Million $)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '4_funding_and_scoring.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 4_funding_and_scoring.png")

# ============================================================================
# Requirement 5: Number and Type of Priority Categories
# ============================================================================
print(f"\n{'='*70}")
print("5. PRIORITY CATEGORIES")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a. Overall priority distribution (pie chart)
priority_total = df['Priority_Category'].value_counts()
colors = {'ANGHP': '#e74c3c', 'EO': '#3498db', 'NI': '#2ecc71', 
          'HA': '#f39c12', 'Unknown': '#95a5a6', 'Other': '#95a5a6'}
axes[0, 0].pie(priority_total.values, labels=priority_total.index, 
              autopct='%1.1f%%',
              colors=[colors.get(p, 'gray') for p in priority_total.index],
              startangle=90)
axes[0, 0].set_title('Overall Priority Distribution', 
                    fontsize=12, fontweight='bold')

# 5b. Priority categories by year
priority_by_year = df.groupby(['Year', 'Priority_Category']).size().unstack(fill_value=0)
priority_by_year.plot(kind='bar', stacked=True, ax=axes[0, 1],
                     color=[colors.get(p, 'gray') for p in priority_by_year.columns])
axes[0, 1].set_title('Priority Categories by Year', 
                    fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Number of Applications')
axes[0, 1].legend(title='Priority', fontsize=9)
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 5c. Priority by application type
priority_by_type = df.groupby(['App_Type', 'Priority_Category']).size().unstack(fill_value=0)
priority_by_type.plot(kind='bar', ax=axes[1, 0],
                     color=[colors.get(p, 'gray') for p in priority_by_type.columns])
axes[1, 0].set_title('Priority Distribution by Application Type', 
                    fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Application Type')
axes[1, 0].set_ylabel('Number of Applications')
axes[1, 0].legend(title='Priority', fontsize=9)
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=15, ha='right')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 5d. Priority trends over time (area chart)
main_priorities = [p for p in ['ANGHP', 'EO', 'NI', 'HA'] if p in priority_by_year.columns]
priority_by_year[main_priorities].plot(kind='area', stacked=True, ax=axes[1, 1], 
                                       alpha=0.7,
                                       color=[colors[p] for p in main_priorities])
axes[1, 1].set_title('Priority Trends Over Time', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Number of Applications')
axes[1, 1].legend(title='Priority', fontsize=9)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '5_priority_categories.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 5_priority_categories.png")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*70}")
print("VISUALIZATION COMPLETE")
print(f"{'='*70}")
print(f"\nGenerated files:")
print(f"  1. 1_applications_per_year.png")
print(f"  2. 2_score_spread.png")
print(f"  3. 3_organizations.png")
print(f"  4. 4_funding_and_scoring.png")
print(f"  5. 5_priority_categories.png")
print(f"\nAll figures saved to: {OUTPUT_DIR}/")
print("="*70)
