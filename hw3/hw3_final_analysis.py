"""
Final Complete Analysis
Automatically detects column names and generates all report assets

Output:
- Charts: output/report_assets/charts/ (5 PNG files)
- Tables: output/report_assets/tables/ (8 CSV files)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# ============================================================================
# Configuration
# ============================================================================
# Try multiple possible file names
POSSIBLE_DATA_FILES = [Path("hw3_output/cleaned_data_2022_2026.csv")]

DATA_FILE = None
for path in POSSIBLE_DATA_FILES:
    if path.exists():
        DATA_FILE = path
        break

if DATA_FILE is None:
    print("ERROR: No cleaned data file found!")
    print("Please run hw3_data_cleaning.py first")
    print("\nSearched for:")
    for p in POSSIBLE_DATA_FILES:
        print(f"  - {p}")
    sys.exit(1)

OUTPUT_DIR = Path("hw3_output/report_assets")
CHARTS_DIR = OUTPUT_DIR / "charts"
TABLES_DIR = OUTPUT_DIR / "tables"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

print("="*70)
print("HW3 FINAL ANALYSIS - Following 2021 PDF Structure")
print("="*70)
print(f"Using data: {DATA_FILE}")

# ============================================================================
# Load and Validate Data
# ============================================================================
df = pd.read_csv(DATA_FILE)

# Auto-detect column names (handle variations)
col_map = {}
for col in df.columns:
    col_lower = col.lower()
    if 'year' in col_lower:
        col_map['Year'] = col
    elif 'organization' in col_lower:
        col_map['Organization'] = col
    elif 'request' in col_lower or col == 'Request':
        col_map['Request'] = col
    elif 'award' in col_lower or col == 'Award':
        col_map['Award'] = col
    elif 'total' in col_lower and 'score' in col_lower or col == 'Total_Score':
        col_map['Total_Score'] = col
    elif 'app' in col_lower and 'type' in col_lower or col == 'App_Type':
        col_map['App_Type'] = col
    elif 'priority' in col_lower and 'category' in col_lower or col == 'Priority_Category':
        col_map['Priority_Category'] = col
    elif 'impact' in col_lower and 'score' in col_lower or col == 'Score_Impact':
        col_map['Score_Impact'] = col
    elif 'principle' in col_lower and 'score' in col_lower or col == 'Score_Principles':
        col_map['Score_Principles'] = col
    elif 'capacity' in col_lower and 'score' in col_lower or col == 'Score_Capacity':
        col_map['Score_Capacity'] = col
    elif 'collab' in col_lower and 'score' in col_lower or col == 'Score_Collab':
        col_map['Score_Collab'] = col

print(f"\nDetected columns:")
for key, val in col_map.items():
    print(f"  {key}: {val}")

# Rename columns for consistency
df = df.rename(columns={
    col_map.get('Request', 'Request'): 'Request',
    col_map.get('Award', 'Award'): 'Award',
    col_map.get('Total_Score', 'Total_Score'): 'Total_Score',
    col_map.get('App_Type', 'App_Type'): 'App_Type',
    col_map.get('Priority_Category', 'Priority_Category'): 'Priority_Category',
})

print(f"\nData loaded: {len(df)} records")
print(f"Years: {sorted(df['Year'].unique())}")

# Separate by category
df_ss = df[df['App_Type'] == 'Social Services'].copy()
df_con = df[df['App_Type'] == 'Construction/Development'].copy()

print(f"\nSocial Services: {len(df_ss)} records")
print(f"Construction/Development: {len(df_con)} records")

years = sorted(df['Year'].unique())
years_labels = [str(int(y)) for y in years]

# ============================================================================
# PAGE 2: Summary Metrics (CHART)
# ============================================================================
print(f"\n{'='*70}")
print("Generating Page 2: Summary Metrics")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Total applications
apps_by_year = df.groupby('Year').size()
axes[0, 0].bar(range(len(years)), apps_by_year.values, color='#2E5090', width=0.6)
axes[0, 0].set_xticks(range(len(years)))
axes[0, 0].set_xticklabels(years_labels, fontsize=12)
axes[0, 0].set_title('# of Applications', fontsize=16, fontweight='bold', pad=15)
axes[0, 0].set_ylabel('Number of Applications', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(apps_by_year.values):
    axes[0, 0].text(i, val + 0.5, str(int(val)), ha='center', fontweight='bold', fontsize=14)

# By category
apps_by_cat = df.groupby(['Year', 'App_Type']).size().unstack(fill_value=0)
x = np.arange(len(years))
width = 0.35
axes[0, 1].bar(x - width/2, apps_by_cat['Social Services'], width, 
              label='Social Services', color='#2E5090')
axes[0, 1].bar(x + width/2, apps_by_cat['Construction/Development'], width, 
              label='Construction/Development', color='#F4B183')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(years_labels, fontsize=12)
axes[0, 1].set_title('Applications by Category', fontsize=16, fontweight='bold', pad=15)
axes[0, 1].set_ylabel('Number of Applications', fontsize=12)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# SS avg score
avg_score_ss = df_ss.groupby('Year')['Total_Score'].mean()
axes[1, 0].bar(range(len(avg_score_ss)), avg_score_ss.values, color='#2E5090', width=0.6)
axes[1, 0].set_xticks(range(len(avg_score_ss)))
axes[1, 0].set_xticklabels([str(int(y)) for y in avg_score_ss.index], fontsize=12)
axes[1, 0].set_title('AVG Score - Social Services', fontsize=16, fontweight='bold', pad=15)
axes[1, 0].set_ylabel('Average Score', fontsize=12)
axes[1, 0].set_ylim([80, 95])
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(avg_score_ss.values):
    axes[1, 0].text(i, val + 0.7, f'{val:.1f}', ha='center', fontweight='bold', fontsize=13)

# CON avg score
avg_score_con = df_con.groupby('Year')['Total_Score'].mean()
axes[1, 1].bar(range(len(avg_score_con)), avg_score_con.values, color='#F4B183', width=0.6)
axes[1, 1].set_xticks(range(len(avg_score_con)))
axes[1, 1].set_xticklabels([str(int(y)) for y in avg_score_con.index], fontsize=12)
axes[1, 1].set_title('AVG Score - Construction/Development', fontsize=16, fontweight='bold', pad=15)
axes[1, 1].set_ylabel('Average Score', fontsize=12)
axes[1, 1].set_ylim([70, 85])
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(avg_score_con.values):
    axes[1, 1].text(i, val + 0.7, f'{val:.1f}', ha='center', fontweight='bold', fontsize=13)

plt.suptitle('Summary Metrics By Year', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(CHARTS_DIR / 'page2_summary_metrics.png', dpi=300, bbox_inches='tight')
print("✓ page2_summary_metrics.png")
plt.close()

# ============================================================================
# PAGE 3: Applicants (TABLES)
# ============================================================================
print(f"\n{'='*70}")
print("Generating Page 3: Applicants Tables")
print(f"{'='*70}")

# SS applicants
applicants_ss = df_ss.groupby(['Organization', 'Year']).size().unstack(fill_value=0)
applicants_ss['Total'] = applicants_ss.sum(axis=1)
applicants_ss = applicants_ss.sort_values('Total', ascending=False).head(15).reset_index()
applicants_ss.columns = ['Organization'] + [str(int(c)) if c != 'Total' else c for c in applicants_ss.columns[1:]]
applicants_ss.to_csv(TABLES_DIR / 'page3_applicants_ss.csv', index=False)
print("✓ page3_applicants_ss.csv")

# CON applicants
applicants_con = df_con.groupby(['Organization', 'Year']).size().unstack(fill_value=0)
applicants_con['Total'] = applicants_con.sum(axis=1)
applicants_con = applicants_con.sort_values('Total', ascending=False).head(15).reset_index()
applicants_con.columns = ['Organization'] + [str(int(c)) if c != 'Total' else c for c in applicants_con.columns[1:]]
applicants_con.to_csv(TABLES_DIR / 'page3_applicants_con.csv', index=False)
print("✓ page3_applicants_con.csv")

# ============================================================================
# PAGE 4: Funded (TABLES)
# ============================================================================
print(f"\n{'='*70}")
print("Generating Page 4: Funded Tables")
print(f"{'='*70}")

# SS funded
funded_ss_by_year = df_ss[df_ss['Award'].notna()].groupby(['Organization', 'Year']).size().unstack(fill_value=0)
total_apps_ss = df_ss.groupby('Organization').size()
total_funded_ss = df_ss[df_ss['Award'].notna()].groupby('Organization').size()

funded_ss = funded_ss_by_year.copy()
funded_ss['Total_Funded'] = total_funded_ss
funded_ss['Total_Apps'] = total_apps_ss
funded_ss['Fund_%'] = (funded_ss['Total_Funded'] / funded_ss['Total_Apps'] * 100).round(0).astype(int)
funded_ss = funded_ss.sort_values('Total_Apps', ascending=False).head(15).reset_index()
funded_ss.columns = ['Organization'] + [str(int(c)) if isinstance(c, (int, float)) and c not in ['Total_Funded', 'Total_Apps', 'Fund_%'] else c for c in funded_ss.columns[1:]]
funded_ss.to_csv(TABLES_DIR / 'page4_funded_ss.csv', index=False)
print("✓ page4_funded_ss.csv")

# CON funded
funded_con_by_year = df_con[df_con['Award'].notna()].groupby(['Organization', 'Year']).size().unstack(fill_value=0)
total_apps_con = df_con.groupby('Organization').size()
total_funded_con = df_con[df_con['Award'].notna()].groupby('Organization').size()

funded_con = funded_con_by_year.copy()
funded_con['Total_Funded'] = total_funded_con
funded_con['Total_Apps'] = total_apps_con
funded_con['Fund_%'] = (funded_con['Total_Funded'] / funded_con['Total_Apps'] * 100).round(0).astype(int)
funded_con = funded_con.sort_values('Total_Apps', ascending=False).head(10).reset_index()
funded_con.columns = ['Organization'] + [str(int(c)) if isinstance(c, (int, float)) and c not in ['Total_Funded', 'Total_Apps', 'Fund_%'] else c for c in funded_con.columns[1:]]
funded_con.to_csv(TABLES_DIR / 'page4_funded_con.csv', index=False)
print("✓ page4_funded_con.csv")

# ============================================================================
# PAGE 5: Priority (TABLES)
# ============================================================================
print(f"\n{'='*70}")
print("Generating Page 5: Priority Tables")
print(f"{'='*70}")

# SS priority
priority_ss_app = df_ss.groupby(['Priority_Category', 'Year']).size().unstack(fill_value=0)
priority_ss_fund = df_ss[df_ss['Award'].notna()].groupby(['Priority_Category', 'Year']).size().unstack(fill_value=0)

priority_ss_app['Total_Applied'] = priority_ss_app.sum(axis=1)
priority_ss_fund['Total_Funded'] = priority_ss_fund.sum(axis=1)

priority_ss_table = priority_ss_app.copy()
priority_ss_table['Total_Funded'] = priority_ss_fund['Total_Funded']
priority_ss_table['Fund_%'] = (priority_ss_table['Total_Funded'] / priority_ss_table['Total_Applied'] * 100).round(0).astype(int)
priority_ss_table = priority_ss_table.reset_index()
priority_ss_table.columns = ['Priority'] + [str(int(c)) if isinstance(c, (int, float)) and c not in ['Total_Applied', 'Total_Funded', 'Fund_%'] else c for c in priority_ss_table.columns[1:]]
priority_ss_table.to_csv(TABLES_DIR / 'page5_priority_ss.csv', index=False)
print("✓ page5_priority_ss.csv")

# CON priority
priority_con_app = df_con.groupby(['Priority_Category', 'Year']).size().unstack(fill_value=0)
priority_con_fund = df_con[df_con['Award'].notna()].groupby(['Priority_Category', 'Year']).size().unstack(fill_value=0)

priority_con_app['Total_Applied'] = priority_con_app.sum(axis=1)
priority_con_fund['Total_Funded'] = priority_con_fund.sum(axis=1)

priority_con_table = priority_con_app.copy()
priority_con_table['Total_Funded'] = priority_con_fund['Total_Funded']
priority_con_table['Fund_%'] = (priority_con_table['Total_Funded'] / priority_con_table['Total_Applied'] * 100).round(0).astype(int)
priority_con_table = priority_con_table.reset_index()
priority_con_table.columns = ['Priority'] + [str(int(c)) if isinstance(c, (int, float)) and c not in ['Total_Applied', 'Total_Funded', 'Fund_%'] else c for c in priority_con_table.columns[1:]]
priority_con_table.to_csv(TABLES_DIR / 'page5_priority_con.csv', index=False)
print("✓ page5_priority_con.csv")

# ============================================================================
# PAGE 6: Funding Distribution (HEATMAP)
# ============================================================================
print(f"\n{'='*70}")
print("Generating Page 6: Funding Distribution Heatmap")
print(f"{'='*70}")

bins = [0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 200000, 500000]
labels = ['0-20k', '20-40k', '40-60k', '60-80k', '80-100k', '100-120k', '120-140k', '140-200k', '200k+']

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# SS heatmap
df_ss['Funding_Range'] = pd.cut(df_ss['Request'], bins=bins, labels=labels)
funding_ss = df_ss.groupby(['Year', 'Funding_Range'], observed=True).size().unstack(fill_value=0)
funding_ss_display = funding_ss.T
funding_ss_display.columns = [str(int(c)) for c in funding_ss_display.columns]

sns.heatmap(funding_ss_display, annot=True, fmt='g', cmap='YlOrRd', ax=axes[0], 
           cbar_kws={'label': '# of Projects'}, linewidths=0.5)
axes[0].set_title('Project Size Distribution - Social Services', fontsize=14, fontweight='bold', pad=10)
axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_ylabel('Funding Request Range', fontsize=12)

# CON heatmap
df_con['Funding_Range'] = pd.cut(df_con['Request'], bins=bins, labels=labels)
funding_con = df_con.groupby(['Year', 'Funding_Range'], observed=True).size().unstack(fill_value=0)
funding_con_display = funding_con.T
funding_con_display.columns = [str(int(c)) for c in funding_con_display.columns]

sns.heatmap(funding_con_display, annot=True, fmt='g', cmap='YlGnBu', ax=axes[1], 
           cbar_kws={'label': '# of Projects'}, linewidths=0.5)
axes[1].set_title('Project Size Distribution - Construction/Development', 
                 fontsize=14, fontweight='bold', pad=10)
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_ylabel('Funding Request Range', fontsize=12)

plt.tight_layout()
plt.savefig(CHARTS_DIR / 'page6_funding_distribution.png', dpi=300, bbox_inches='tight')
print("✓ page6_funding_distribution.png")
plt.close()

# ============================================================================
# PAGE 7: Score vs Funding (SCATTER)
# ============================================================================
print(f"\n{'='*70}")
print("Generating Page 7: Score vs Funding Scatter")
print(f"{'='*70}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# SS scatter
df_ss_scores = df_ss[df_ss['Total_Score'].notna()]
scatter1 = axes[0].scatter(df_ss_scores['Request'], df_ss_scores['Total_Score'], 
                          alpha=0.6, s=120, c=df_ss_scores['Year'], cmap='viridis', 
                          edgecolors='black', linewidth=0.5)
axes[0].set_title('Project Score by Funding Request - Social Services', 
                 fontsize=13, fontweight='bold')
axes[0].set_xlabel('Funding Request ($)', fontsize=11)
axes[0].set_ylabel('Project Score', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([75, 100])
cbar1 = plt.colorbar(scatter1, ax=axes[0])
cbar1.set_label('Year', fontsize=10)
cbar1.set_ticks([2022, 2023, 2024, 2025])

# CON scatter
df_con_scores = df_con[df_con['Total_Score'].notna()]
scatter2 = axes[1].scatter(df_con_scores['Request'], df_con_scores['Total_Score'], 
                          alpha=0.6, s=120, c=df_con_scores['Year'], cmap='plasma',
                          edgecolors='black', linewidth=0.5)
axes[1].set_title('Project Score by Funding Request - Construction/Development', 
                 fontsize=13, fontweight='bold')
axes[1].set_xlabel('Funding Request ($)', fontsize=11)
axes[1].set_ylabel('Project Score', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([60, 95])
cbar2 = plt.colorbar(scatter2, ax=axes[1])
cbar2.set_label('Year', fontsize=10)
cbar2.set_ticks([2022, 2023, 2024, 2025])

plt.tight_layout()
plt.savefig(CHARTS_DIR / 'page7_score_vs_funding.png', dpi=300, bbox_inches='tight')
print("✓ page7_score_vs_funding.png")
plt.close()

# ============================================================================
# PAGE 8: Score Distribution (STACKED BAR)
# ============================================================================
print(f"\n{'='*70}")
print("Generating Page 8: Score Distribution")
print(f"{'='*70}")

score_bins = [0, 75, 80, 85, 90, 95, 100]
score_labels = ['0-75', '75-80', '80-85', '85-90', '90-95', '95-100']

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# SS stacked
df_ss['Score_Range'] = pd.cut(df_ss['Total_Score'], bins=score_bins, labels=score_labels)
score_dist_ss = df_ss.groupby(['Year', 'Score_Range'], observed=True).size().unstack(fill_value=0)

score_dist_ss.plot(kind='bar', stacked=True, ax=axes[0], 
                  color=['#8B0000', '#CD5C5C', '#F0E68C', '#90EE90', '#228B22', '#006400'],
                  width=0.7)
axes[0].set_title('Score Distribution by Year - Social Services', 
                 fontsize=14, fontweight='bold', pad=10)
axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_ylabel('# of Projects at Score Range', fontsize=12)
axes[0].legend(title='Score Range', fontsize=9, loc='upper left')
axes[0].set_xticklabels([str(int(y)) for y in score_dist_ss.index], rotation=0, fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# CON stacked
df_con['Score_Range'] = pd.cut(df_con['Total_Score'], bins=score_bins, labels=score_labels)
score_dist_con = df_con.groupby(['Year', 'Score_Range'], observed=True).size().unstack(fill_value=0)

score_dist_con.plot(kind='bar', stacked=True, ax=axes[1], 
                   color=['#8B0000', '#CD5C5C', '#F0E68C', '#90EE90', '#228B22', '#006400'],
                   width=0.7)
axes[1].set_title('Score Distribution by Year - Construction/Development', 
                 fontsize=14, fontweight='bold', pad=10)
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_ylabel('# of Projects at Score Range', fontsize=12)
axes[1].legend(title='Score Range', fontsize=9, loc='upper left')
axes[1].set_xticklabels([str(int(y)) for y in score_dist_con.index], rotation=0, fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(CHARTS_DIR / 'page8_score_distribution.png', dpi=300, bbox_inches='tight')
print("✓ page8_score_distribution.png")
plt.close()

# ============================================================================
# PAGE 9: Scoring Breakdown (TABLE + CHART)
# ============================================================================
print(f"\n{'='*70}")
print("Generating Page 9: Scoring Breakdown")
print(f"{'='*70}")

# Check if scoring breakdown columns exist
score_cols = ['Score_Impact', 'Score_Principles', 'Score_Capacity', 'Score_Collab']
has_breakdown = all(col in df.columns for col in score_cols)

if has_breakdown:
    df_breakdown = df[df['Score_Impact'].notna()].copy()
    
    if len(df_breakdown) > 0:
        # Create table
        max_points = [30, 30, 25, 15]
        breakdown_data = []
        
        for i, col in enumerate(score_cols):
            ss_avg = df_breakdown[df_breakdown['App_Type'] == 'Social Services'][col].mean()
            ss_pct = (ss_avg / max_points[i] * 100).round(0)
            con_avg = df_breakdown[df_breakdown['App_Type'] == 'Construction/Development'][col].mean()
            con_pct = (con_avg / max_points[i] * 100).round(0)
            
            breakdown_data.append({
                'Category': col.replace('Score_', ''),
                'Max_Points': max_points[i],
                'SS_Avg': round(ss_avg, 1),
                'SS_%': int(ss_pct),
                'CON_Avg': round(con_avg, 1) if not np.isnan(con_avg) else 0,
                'CON_%': int(con_pct) if not np.isnan(con_pct) else 0
            })
        
        breakdown_table = pd.DataFrame(breakdown_data)
        breakdown_table.to_csv(TABLES_DIR / 'page9_scoring_breakdown.csv', index=False)
        print("✓ page9_scoring_breakdown.csv")
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(score_cols))
        width = 0.35
        
        labels = ['Priority\nImpact\n(30 pts)', 'Guiding\nPrinciples\n(30 pts)', 
                 'Capacity\nto Deliver\n(25 pts)', 'Collaboration\nPartnership\n(15 pts)']
        
        ss_scores = breakdown_table['SS_Avg'].values
        con_scores = breakdown_table['CON_Avg'].values
        
        ax.bar(x - width/2, ss_scores, width, label='Social Services', color='#2E5090')
        ax.bar(x + width/2, con_scores, width, label='Construction/Development', color='#F4B183')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title('Scoring Breakdown by Category', fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 32])
        
        for i in range(len(score_cols)):
            ax.text(i - width/2, ss_scores[i] + 0.5, f'{ss_scores[i]:.1f}', 
                   ha='center', fontweight='bold', fontsize=10)
            if con_scores[i] > 0:
                ax.text(i + width/2, con_scores[i] + 0.5, f'{con_scores[i]:.1f}', 
                       ha='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / 'page9_scoring_breakdown.png', dpi=300, bbox_inches='tight')
        print("✓ page9_scoring_breakdown.png")
        plt.close()
    else:
        print("  ⚠ No scoring breakdown data available")
else:
    print("  ⚠ Scoring breakdown columns not found in data")
    print(f"  Available columns: {df.columns.tolist()}")

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*70}")
print("GENERATION COMPLETE")
print(f"{'='*70}")
print(f"\n📊 CHARTS generated in: {CHARTS_DIR}/")
print(f"  - page2_summary_metrics.png")
print(f"  - page6_funding_distribution.png")
print(f"  - page7_score_vs_funding.png")
print(f"  - page8_score_distribution.png")
if has_breakdown:
    print(f"  - page9_scoring_breakdown.png")

print(f"\n📋 TABLES generated in: {TABLES_DIR}/")
print(f"  - page3_applicants_ss.csv / page3_applicants_con.csv")
print(f"  - page4_funded_ss.csv / page4_funded_con.csv")
print(f"  - page5_priority_ss.csv / page5_priority_con.csv")
if has_breakdown:
    print(f"  - page9_scoring_breakdown.csv")

print(f"\n✅ All years display as integers (2022, 2023, 2024, 2025)")
print(f"✅ All analyses separated by Social Services vs Construction/Development")
print(f"✅ Ready for PowerPoint report")
print("="*70)
