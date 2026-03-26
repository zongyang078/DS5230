"""
Compares trends from 2016-2022 period vs 2022-2026 period
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
DATA_2016_2022 = Path("data/2016-2022 Case Data.xlsx")
OUTPUT_DIR = Path("hw3_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# Helper Functions (reuse from main script)
# ============================================================================

def to_number(value):
    """Convert string to numeric"""
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except:
        return np.nan


def extract_year(text):
    """Extract year from text"""
    match = re.search(r"(20\d{2})", str(text))
    return int(match.group(1)) if match else np.nan


def classify_app_type(text):
    """Classify application type"""
    s = str(text).strip().lower()
    if "social" in s or s == "ss":
        return "Social Services"
    if any(x in s for x in ["con", "dev", "econ"]):
        return "Construction/Development"
    return "Other"


def classify_priority(text):
    """Classify priority (including old codes BN, SN, WS)"""
    s = str(text).strip().upper()
    if s in ("NAN", "NONE", "", "ALL"):
        return "Unknown"
    # Current priorities
    if "HOMELESS" in s or "ANGHP" in s:
        return "ANGHP"
    if "ECONOMIC" in s or "EO" in s:
        return "EO"
    if "NEIGHBORHOOD" in s or "NI" in s:
        return "NI"
    if "HOUSING" in s or "HA" in s:
        return "HA"
    # Old priority codes (2015-2016 period)
    if "BN" in s:
        return "BN"  # Basic Needs
    if "SN" in s:
        return "SN"  # Special Needs  
    if "WS" in s:
        return "WS"  # Workforce Support
    return "Other"


def is_summary_row(org):
    """Check if row is a summary"""
    if pd.isna(org):
        return True
    s = str(org).lower()
    return bool(re.search(r"total|subtotal|available|cap|estimated", s))


# ============================================================================
# Process 2016-2022 Data
# ============================================================================

def clean_2016_2022_sheet(sheet_name, file_path):
    """
    Clean a sheet from 2016-2022 data file
    
    Args:
        sheet_name: Name of worksheet
        file_path: Path to Excel file
        
    Returns:
        DataFrame with cleaned data
    """
    print(f"\nProcessing: {sheet_name}")
    
    # 2016-2022 format typically uses header row 1
    header = 1
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
    df = df.dropna(how="all")
    year = extract_year(sheet_name)
    
    cols = df.columns
    
    # Identify columns
    col_idx = {
        "Type": next((i for i, c in enumerate(cols) if "type" in str(c).lower()), 1),
        "Priority": next((i for i, c in enumerate(cols) if 
                         "priority" in str(c).lower() and "impact" not in str(c).lower()), 2),
        "Org": next((i for i, c in enumerate(cols) if "organization" in str(c).lower()), 3),
        "Project": next((i for i, c in enumerate(cols) if 
                        "project" in str(c).lower() or "program" in str(c).lower()), 4),
        "Request": next((i for i, c in enumerate(cols) if "request" in str(c).lower()), 5),
        "Award": len(cols) - 1,
        "Total_Score": next((i for i, c in enumerate(cols) if 
                            "total" in str(c).lower() or "score" in str(c).lower()), None)
    }
    
    # Extract data
    records = []
    for _, row in df.iterrows():
        org = row.iloc[col_idx["Org"]]
        
        if is_summary_row(org):
            continue
        
        records.append({
            "Year": year,
            "Organization": str(org).strip(),
            "Project": row.iloc[col_idx["Project"]],
            "Type": row.iloc[col_idx["Type"]],
            "Priority": row.iloc[col_idx["Priority"]],
            "Funding_Request": to_number(row.iloc[col_idx["Request"]]),
            "Funding_Award": to_number(row.iloc[col_idx["Award"]]),
            "Total_Score": to_number(row.iloc[col_idx["Total_Score"]]) if col_idx["Total_Score"] else np.nan,
        })
    
    result = pd.DataFrame(records)
    result["App_Type"] = result["Type"].apply(classify_app_type)
    result["Priority_Category"] = result["Priority"].apply(classify_priority)
    
    print(f"  Extracted {len(result)} records")
    return result


def main():
    """Main function: clean 2016-2022 data"""
    
    print("="*70)
    print("STRETCH GOAL: Cleaning 2016-2022 Data")
    print("="*70)
    
    xlsx = pd.ExcelFile(DATA_2016_2022)
    print(f"\nWorksheets: {xlsx.sheet_names}")
    
    all_data = [clean_2016_2022_sheet(sheet, DATA_2016_2022) 
                for sheet in xlsx.sheet_names]
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Filter: keep only Social Services and Construction/Development
    combined = combined[
        combined["App_Type"].isin(["Social Services", "Construction/Development"])
    ].copy()
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL DATASET (2016-2022)")
    print(f"{'='*70}")
    print(f"Total records: {len(combined)}")
    print(f"\nBy Year:")
    print(combined['Year'].value_counts().sort_index())
    print(f"\nBy Category:")
    print(combined['App_Type'].value_counts())
    print(f"\nBy Priority:")
    print(combined['Priority_Category'].value_counts())
    
    # Save
    output = OUTPUT_DIR / "cleaned_data_2016_2022.csv"
    combined.to_csv(output, index=False)
    
    print(f"\n✓ Saved to: {output}")
    print("="*70)
    
    return combined


if __name__ == "__main__":
    df = main()
