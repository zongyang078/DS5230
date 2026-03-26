"""
Part 3: Apriori Algorithm on a New Dataset
Dataset: Groceries_dataset.csv (Kaggle)
- 38,765 purchase records from a grocery store
- Each row: Member_number, Date, itemDescription
- Grouped by (Member_number, Date) into transactions
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from collections import defaultdict
import csv



# 1. Load & Group into Transactions

def load_transactions(filepath):
    """
    Each row is (Member_number, Date, itemDescription).
    Group by (Member_number, Date) to form transactions.
    """
    txn_dict = defaultdict(set)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['Member_number'].strip(), row['Date'].strip())
            txn_dict[key].add(row['itemDescription'].strip())

    transactions = [list(items) for items in txn_dict.values() if len(items) > 1]
    return transactions



# 2. Encode for mlxtend

def encode_transactions(transactions):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)
    return df



# 3. Helper: print rules

def print_top_rules(rules, top_n=10, sort_by='lift'):
    if rules.empty:
        print("    No rules found.")
        return
    rules_sorted = rules.sort_values(sort_by, ascending=False).head(top_n)
    for _, r in rules_sorted.iterrows():
        ant = ", ".join(sorted(r['antecedents']))
        con = ", ".join(sorted(r['consequents']))
        print(f"    {{{ant}}} => {{{con}:<25s}} "
              f"sup={r['support']:.4f}  conf={r['confidence']:.4f}  lift={r['lift']:.2f}")



# 4. Main

if __name__ == "__main__":
    filepath = "data/Groceries_dataset.csv"

    print("=" * 80)
    print("  PART 3: Apriori on Kaggle Groceries Dataset")
    print("=" * 80)

    # Load
    transactions = load_transactions(filepath)
    print(f"\nTransactions (with 2+ items): {len(transactions)}")

    all_items = set()
    for t in transactions:
        all_items.update(t)
    print(f"Unique items: {len(all_items)}")

    sizes = [len(t) for t in transactions]
    print(f"Items per transaction: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")

    # Encode
    df = encode_transactions(transactions)
    print(f"Encoded DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")

    # ─── Experiment 1: Vary min_support ────────────────────
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1: Varying Minimum Support (fixed confidence = 0.3)")
    print("=" * 80)

    support_values = [0.005, 0.01, 0.02, 0.03, 0.05]
    for ms in support_values:
        freq = apriori(df, min_support=ms, use_colnames=True)
        if freq.empty:
            print(f"\n  min_support = {ms} ({ms*100:.1f}%): No frequent itemsets")
            continue
        rules = association_rules(freq, metric="confidence", min_threshold=0.3)
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

    # ─── Detailed Output ──────────────────────────────────
    print("\n" + "=" * 80)
    print("  DETAILED RESULTS: min_support=0.01, min_confidence=0.3")
    print("=" * 80)

    freq = apriori(df, min_support=0.01, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.3)
    rules_sorted = rules.sort_values('lift', ascending=False)

    print(f"\n  Total frequent itemsets: {len(freq)}")
    for k in range(1, 5):
        cnt = len(freq[freq['itemsets'].apply(len) == k])
        if cnt > 0:
            print(f"    L{k}: {cnt} frequent {k}-itemsets")

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

    # ─── Top 10 most purchased items ───────────────────────
    print("\n" + "=" * 80)
    print("  TOP 10 MOST PURCHASED ITEMS")
    print("=" * 80)

    singles = freq[freq['itemsets'].apply(len) == 1].sort_values('support', ascending=False)
    for _, row in singles.head(10).iterrows():
        item = list(row['itemsets'])[0]
        print(f"  {item}: support = {row['support']:.4f} ({row['support']*100:.2f}%)")