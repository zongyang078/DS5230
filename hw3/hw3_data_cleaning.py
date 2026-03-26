"""
Cleans CDBG application data for 2022-2026 analysis
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Configuration
DATA_FILE = Path("data/2022-2026 Case Data.xlsx")
OUTPUT_DIR = Path("hw3_output")
OUTPUT_DIR.mkdir(exist_ok=True)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


# Helper Functions
def to_number(value):
    """Convert string to numeric, handling currency symbols and commas"""
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except:
        return np.nan


def extract_year(text):
    """Extract first year from text (e.g., '2022-2023' -> 2022)"""
    match = re.search(r"(20\d{2})", str(text))
    return int(match.group(1)) if match else np.nan


def classify_app_type(text):
    """Classify application into standard categories"""
    s = str(text).strip().lower()
    if "social" in s or s == "ss":
        return "Social Services"
    if any(x in s for x in ["con", "dev", "econ"]):
        return "Construction/Development"
    if "admin" in s or s == "ap":
        return "Admin"
    if "planning" in s:
        return "Planning"
    return "Other"


def classify_priority(text):
    """Map priority to standard categories (ANGHP, EO, NI, HA)"""
    s = str(text).strip().upper()
    if s in ("NAN", "NONE", "", "ALL"):
        return "Unknown"
    if "HOMELESS" in s or "ANGHP" in s:
        return "ANGHP"  # Addressing Needs of Growing Homeless Population
    if "ECONOMIC" in s or "EO" in s:
        return "EO"  # Economic Opportunity
    if "NEIGHBORHOOD" in s or "NI" in s:
        return "NI"  # Neighborhood Investment and Infrastructure
    if "HOUSING" in s or "HA" in s:
        return "HA"  # Housing Availability
    return "Other"


def is_summary_row(org):
    """Check if row is a summary/total row (not actual application data)"""
    if pd.isna(org):
        return True
    s = str(org).lower()
    return bool(re.search(r"total|subtotal|available|cap|estimated", s))


# Main Processing Function
def clean_sheet(sheet_name, file_path):
    """
    Clean a single worksheet from the Excel file
    
    Args:
        sheet_name: Name of the worksheet
        file_path: Path to Excel file
        
    Returns:
        DataFrame with cleaned data
    """
    print(f"\n{'='*70}")
    print(f"Processing: {sheet_name}")
    print(f"{'='*70}")
    
    # Step 1: Determine header row location
    # 2022-23 and 2023-24 use row 2, 2024-25/2025-26 need to search
    if any(yr in sheet_name for yr in ["2022", "2023"]):
        header = 2
    else:
        # Search for row containing "Organization" or "Case Id"
        df_temp = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        header = next((i for i in range(20) if 
                      "organization" in str(df_temp.iloc[i]).lower()), 8)
    
    # Step 2: Read data with identified header
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
    df = df.dropna(how="all")  # Remove completely empty rows
    year = extract_year(sheet_name)
    
    cols = df.columns
    print(f"Columns: {len(cols)}")
    
    # Step 3: Identify key columns by searching for keywords
    col_idx = {
        "Type": next((i for i, c in enumerate(cols) if "type" in str(c).lower()), 1),
        "Priority": next((i for i, c in enumerate(cols) if 
                         "priority" in str(c).lower() and "impact" not in str(c).lower()), 2),
        "Org": next((i for i, c in enumerate(cols) if "organization" in str(c).lower()), 3),
        "Project": next((i for i, c in enumerate(cols) if 
                        "project" in str(c).lower() or "program" in str(c).lower()), 4),
        "Request": next((i for i, c in enumerate(cols) if "request" in str(c).lower()), 5),
        "Award": len(cols) - 1,  # Last column is typically award
        "Total_Score": next((i for i, c in enumerate(cols) if 
                            "avg" in str(c).lower() and "score" in str(c).lower()), 
                           12 if len(cols) > 12 else None)
    }
    
    # Step 4: Identify scoring breakdown columns
    # These represent the breakdown of scoring sections required by Rowen
    score_breakdown = {
        "Score_Impact": None,       # Priority Impact/Goal (typically 30 pts)
        "Score_Principles": None,   # Guiding Principles (typically 30 pts)
        "Score_Capacity": None,     # Capacity to Deliver (typically 25 pts)
        "Score_Collab": None        # Collaboration/Partnership (typically 15 pts)
    }
    
    # Method 1: Search by column name (works for 2024-25 and 2025-26)
    for i, col in enumerate(cols):
        col_lower = str(col).lower()
        if "priority" in col_lower and "impact" in col_lower:
            score_breakdown["Score_Impact"] = i
        elif "guiding" in col_lower and "principle" in col_lower:
            score_breakdown["Score_Principles"] = i
        elif "capacity" in col_lower and "deliver" in col_lower:
            score_breakdown["Score_Capacity"] = i
        elif "collaboration" in col_lower or "partnership" in col_lower:
            score_breakdown["Score_Collab"] = i
    
    # Method 2: Identify by point values (works for 2022-23 and 2023-24)
    # These years use column headers like "30 pts", "30 pts", "25 pts", "15 pts"
    if any(yr in sheet_name for yr in ["2022", "2023"]):
        pts_cols = [(i, str(col)) for i, col in enumerate(cols) 
                    if "pt" in str(col).lower()]
        
        if len(pts_cols) >= 4:
            # Scoring order: Impact(30), Principles(30), Capacity(25), Collab(15)
            score_breakdown["Score_Impact"] = pts_cols[0][0] if "30" in pts_cols[0][1] else None
            score_breakdown["Score_Principles"] = pts_cols[1][0] if "30" in pts_cols[1][1] else None
            score_breakdown["Score_Capacity"] = pts_cols[2][0] if "25" in pts_cols[2][1] else None
            score_breakdown["Score_Collab"] = pts_cols[3][0] if "15" in pts_cols[3][1] else None
    
    print(f"Identified scoring breakdown columns:")
    for key, idx in score_breakdown.items():
        col_name = cols[idx] if idx is not None else "Not found"
        print(f"  {key}: {col_name}")
    
    # Step 5: Extract and clean data
    records = []
    for _, row in df.iterrows():
        org = row.iloc[col_idx["Org"]]
        
        # Skip summary/total rows
        if is_summary_row(org):
            continue
        
        # Extract all fields
        record = {
            "Year": year,
            "Organization": str(org).strip(),
            "Project": row.iloc[col_idx["Project"]],
            "Type": row.iloc[col_idx["Type"]],
            "Priority": row.iloc[col_idx["Priority"]],
            "Funding_Request": to_number(row.iloc[col_idx["Request"]]),
            "Funding_Award": to_number(row.iloc[col_idx["Award"]]),
            "Total_Score": to_number(row.iloc[col_idx["Total_Score"]]) if col_idx["Total_Score"] else np.nan,
        }
        
        # Add scoring breakdown (required for "breakdown of scoring sections")
        for key, idx in score_breakdown.items():
            record[key] = to_number(row.iloc[idx]) if idx is not None else np.nan
        
        records.append(record)
    
    # Step 6: Create DataFrame and apply classifications
    result = pd.DataFrame(records)
    result["App_Type"] = result["Type"].apply(classify_app_type)
    result["Priority_Category"] = result["Priority"].apply(classify_priority)
    
    print(f"Extracted {len(result)} records")
    print(f"  By category: {result['App_Type'].value_counts().to_dict()}")
    
    # Show scoring breakdown completeness
    for key in score_breakdown.keys():
        non_null = result[key].notna().sum()
        print(f"  {key}: {non_null}/{len(result)} ({non_null/len(result)*100:.1f}%)")
    
    return result


# Main Execution
def main():
    """Main function: process all sheets and combine"""
    
    print("="*70)
    print("HW3 DATA CLEANING - CDBG Applications 2022-2026")
    print("="*70)
    
    # Process all worksheets
    xlsx = pd.ExcelFile(DATA_FILE)
    all_data = [clean_sheet(sheet, DATA_FILE) for sheet in xlsx.sheet_names]
    
    # Combine all years
    combined = pd.concat(all_data, ignore_index=True)
    
    # Filter: Keep only Social Services and Construction/Development
    # (Ignore Admin and Planning as per Rowen's instructions)
    combined = combined[
        combined["App_Type"].isin(["Social Services", "Construction/Development"])
    ].copy()
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total records: {len(combined)}")
    print(f"\nBy Year:")
    print(combined['Year'].value_counts().sort_index())
    print(f"\nBy Category:")
    print(combined['App_Type'].value_counts())
    print(f"\nBy Priority:")
    print(combined['Priority_Category'].value_counts())
    
    # Data quality checks
    print(f"\n{'='*70}")
    print("DATA QUALITY")
    print(f"{'='*70}")
    
    # Check scoring data completeness
    score_cols = ["Score_Impact", "Score_Principles", "Score_Capacity", 
                  "Score_Collab", "Total_Score"]
    print("\nScoring data completeness:")
    for col in score_cols:
        non_null = combined[col].notna().sum()
        print(f"  {col}: {non_null}/{len(combined)} ({non_null/len(combined)*100:.1f}%)")
    
    print(f"\nMissing Awards: {combined['Funding_Award'].isna().sum()}")
    
    # Save cleaned data
    output_file = OUTPUT_DIR / "cleaned_data_2022_2026.csv"
    combined.to_csv(output_file, index=False)
    
    print(f"\n✓ Data saved to: {output_file}")
    print(f"\nColumns in output: {list(combined.columns)}")
    print("="*70)
    
    return combined


if __name__ == "__main__":
    df = main()
