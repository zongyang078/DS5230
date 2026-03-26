"""
Part 2: Apriori Algorithm Using mlxtend Library
Run Apriori on Market_Basket_Optimisation.csv with different
support/confidence cutoffs and compare with Part 1 results.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import csv



# 1. Load Data

def load_transactions(filepath):
    transactions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            items = [item.strip() for item in row if item.strip()]
            if items:
                transactions.append(items)
    print(f"Loaded {len(transactions)} transactions.")
    return transactions



# 2. Encode into DataFrame for mlxtend

def encode_transactions(transactions):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    print(f"Encoded into DataFrame: {df.shape[0]} rows x {df.shape[1]} items")
    return df



# 3. Run Apriori + Generate Rules

def run_apriori(df, min_support=0.01, min_confidence=0.3, min_lift=1.0):
    freq = apriori(df, min_support=min_support, use_colnames=True)
    if freq.empty:
        return freq, pd.DataFrame()
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] >= min_lift]
    return freq, rules


def print_top_rules(rules, top_n=10, sort_by='lift'):
    if rules.empty:
        print("    No rules found.")
        return
    rules_sorted = rules.sort_values(sort_by, ascending=False).head(top_n)
    for _, r in rules_sorted.iterrows():
        ant = ", ".join(sorted(r['antecedents']))
        con = ", ".join(sorted(r['consequents']))
        print(f"    {{{ant}}} => {{{con}:<20s}} "
              f"sup={r['support']:.4f}  conf={r['confidence']:.4f}  lift={r['lift']:.2f}")



# 4. Main

if __name__ == "__main__":
    filepath = "data/Market_Basket_Optimisation.csv"
    transactions = load_transactions(filepath)
    df = encode_transactions(transactions)

    # ─── Experiment 1: Vary min_support ────────────────────
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1: Varying Minimum Support (fixed confidence = 0.3)")
    print("=" * 80)

    support_values = [0.005, 0.01, 0.02, 0.03, 0.05]
    for ms in support_values:
        freq, rules = run_apriori(df, min_support=ms, min_confidence=0.3)
        freq_2plus = freq[freq['itemsets'].apply(len) >= 2]
        print(f"\n  min_support = {ms} ({ms*100:.1f}%)")
        print(f"    Frequent itemsets: {len(freq)} (pairs+: {len(freq_2plus)})")
        print(f"    Rules generated:   {len(rules)}")
        print_top_rules(rules, top_n=5)

    # ─── Experiment 2: Vary min_confidence ─────────────────
    print("\n" + "=" * 80)
    print("  EXPERIMENT 2: Varying Minimum Confidence (fixed support = 0.01)")
    print("=" * 80)

    freq_base = apriori(df, min_support=0.01, use_colnames=True)
    conf_values = [0.2, 0.3, 0.4, 0.5, 0.6]

    for mc in conf_values:
        if freq_base.empty:
            break
        rules = association_rules(freq_base, metric="confidence", min_threshold=mc)
        print(f"\n  min_confidence = {mc} ({mc*100:.0f}%)")
        print(f"    Rules generated: {len(rules)}")
        print_top_rules(rules, top_n=5)

    # ─── Detailed Output: Best Configuration ───────────────
    print("\n" + "=" * 80)
    print("  DETAILED RESULTS: min_support=0.01, min_confidence=0.3")
    print("=" * 80)

    freq, rules = run_apriori(df, min_support=0.01, min_confidence=0.3)
    rules_sorted = rules.sort_values('lift', ascending=False)

    print(f"\n  Total rules: {len(rules_sorted)}\n")
    print(f"  {'Rule':<55} {'Support':>8} {'Confidence':>11} {'Lift':>8}")
    print("  " + "-" * 85)
    for _, r in rules_sorted.head(30).iterrows():
        ant = ", ".join(sorted(r['antecedents']))
        con = ", ".join(sorted(r['consequents']))
        rule_str = f"{{{ant}}} => {{{con}}}"
        print(f"  {rule_str:<55} {r['support']:>8.4f} {r['confidence']:>11.4f} {r['lift']:>8.2f}")

    # ─── Most Commonly Bought Together ─────────────────────
    print("\n" + "=" * 80)
    print("  MOST COMMONLY BOUGHT TOGETHER (Top 15 pairs)")
    print("=" * 80)

    pairs = freq[freq['itemsets'].apply(len) == 2].sort_values('support', ascending=False)
    for _, row in pairs.head(15).iterrows():
        items = sorted(row['itemsets'])
        print(f"  {items[0]} + {items[1]}: support = {row['support']:.4f} "
              f"({row['support']*100:.2f}%)")