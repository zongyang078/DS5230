"""
HW5 - Part 2: Apriori Algorithm Using Library / Different Cutoffs
=================================================================
Since mlxtend/apyori may not be available, this script provides a
clean library-style wrapper around the Apriori algorithm and tests
multiple support/confidence thresholds to explore the dataset.
"""

import csv
from itertools import combinations
from collections import defaultdict


# ──────────────────────────────────────────────────────────
# Reusable Apriori Library Functions
# ──────────────────────────────────────────────────────────
def load_transactions(filepath):
    transactions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            items = [item.strip() for item in row if item.strip()]
            if items:
                transactions.append(set(items))
    return transactions


def _get_L1(transactions, min_sup_count):
    counts = defaultdict(int)
    for t in transactions:
        for item in t:
            counts[frozenset([item])] += 1
    return {k: v for k, v in counts.items() if v >= min_sup_count}


def _apriori_gen(Lk_prev):
    candidates = set()
    itemsets = list(Lk_prev.keys())
    for i in range(len(itemsets)):
        for j in range(i + 1, len(itemsets)):
            union = itemsets[i] | itemsets[j]
            if len(union) == len(itemsets[i]) + 1:
                # Prune
                k = len(union)
                prune = False
                for sub in combinations(union, k - 1):
                    if frozenset(sub) not in Lk_prev:
                        prune = True
                        break
                if not prune:
                    candidates.add(union)
    return candidates


def run_apriori(transactions, min_support=0.01):
    n = len(transactions)
    min_sup_count = int(min_support * n)
    L1 = _get_L1(transactions, min_sup_count)
    all_freq = dict(L1)
    Lk_prev = L1
    k = 2
    while Lk_prev:
        cands = _apriori_gen(Lk_prev)
        if not cands:
            break
        counts = defaultdict(int)
        for t in transactions:
            for c in cands:
                if c.issubset(t):
                    counts[c] += 1
        Lk = {k_: v for k_, v in counts.items() if v >= min_sup_count}
        if not Lk:
            break
        all_freq.update(Lk)
        Lk_prev = Lk
        k += 1
    return all_freq, n


def generate_rules(freq, n, min_confidence=0.3):
    rules = []
    for itemset, sup_count in freq.items():
        if len(itemset) < 2:
            continue
        support = sup_count / n
        for i in range(1, len(itemset)):
            for ant in combinations(itemset, i):
                ant = frozenset(ant)
                con = itemset - ant
                ant_sup = freq.get(ant, 0)
                con_sup = freq.get(con, 0)
                if ant_sup == 0 or con_sup == 0:
                    continue
                confidence = sup_count / ant_sup
                if confidence >= min_confidence:
                    lift = confidence / (con_sup / n)
                    rules.append({
                        'antecedent': ant,
                        'consequent': con,
                        'support': support,
                        'confidence': confidence,
                        'lift': lift
                    })
    return rules


def format_rule(r):
    ant = ", ".join(sorted(r['antecedent']))
    con = ", ".join(sorted(r['consequent']))
    return f"{{{ant}}} => {{{con}}}"


# ──────────────────────────────────────────────────────────
# Main: Experiment with Different Cutoffs
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    filepath = "Market_Basket_Optimisation.csv"
    transactions = load_transactions(filepath)
    n = len(transactions)
    print(f"Loaded {n} transactions.\n")

    # ─── Experiment 1: Vary min_support ────────────────────
    print("=" * 80)
    print("  EXPERIMENT 1: Varying Minimum Support (fixed confidence = 0.3)")
    print("=" * 80)

    support_values = [0.005, 0.01, 0.02, 0.03, 0.05]
    for ms in support_values:
        freq, _ = run_apriori(transactions, min_support=ms)
        rules = generate_rules(freq, n, min_confidence=0.3)
        freq_2plus = {k: v for k, v in freq.items() if len(k) >= 2}
        print(f"\n  min_support = {ms} ({ms*100:.1f}%)")
        print(f"    Frequent itemsets: {len(freq)} (pairs+: {len(freq_2plus)})")
        print(f"    Rules generated:   {len(rules)}")
        if rules:
            top = sorted(rules, key=lambda x: x['lift'], reverse=True)[:5]
            print(f"    Top 5 rules by lift:")
            for r in top:
                print(f"      {format_rule(r):<55} "
                      f"sup={r['support']:.4f}  conf={r['confidence']:.4f}  lift={r['lift']:.2f}")

    # ─── Experiment 2: Vary min_confidence ─────────────────
    print("\n" + "=" * 80)
    print("  EXPERIMENT 2: Varying Minimum Confidence (fixed support = 0.01)")
    print("=" * 80)

    conf_values = [0.2, 0.3, 0.4, 0.5, 0.6]
    freq, _ = run_apriori(transactions, min_support=0.01)

    for mc in conf_values:
        rules = generate_rules(freq, n, min_confidence=mc)
        print(f"\n  min_confidence = {mc} ({mc*100:.0f}%)")
        print(f"    Rules generated: {len(rules)}")
        if rules:
            top = sorted(rules, key=lambda x: x['lift'], reverse=True)[:5]
            print(f"    Top 5 rules by lift:")
            for r in top:
                print(f"      {format_rule(r):<55} "
                      f"sup={r['support']:.4f}  conf={r['confidence']:.4f}  lift={r['lift']:.2f}")

    # ─── Detailed Analysis: Best Configuration ─────────────
    print("\n" + "=" * 80)
    print("  DETAILED ANALYSIS: min_support=0.01, min_confidence=0.3")
    print("=" * 80)

    freq, _ = run_apriori(transactions, min_support=0.01)
    rules = generate_rules(freq, n, min_confidence=0.3)
    rules_sorted = sorted(rules, key=lambda x: x['lift'], reverse=True)

    print(f"\n  Total rules: {len(rules_sorted)}")
    print(f"\n  {'Rule':<55} {'Support':>8} {'Confidence':>11} {'Lift':>8}")
    print("  " + "-" * 85)
    for r in rules_sorted[:30]:
        print(f"  {format_rule(r):<55} {r['support']:>8.4f} {r['confidence']:>11.4f} {r['lift']:>8.2f}")

    # ─── Most Commonly Bought Together ─────────────────────
    print("\n" + "=" * 80)
    print("  MOST COMMONLY BOUGHT TOGETHER (Top 15 pairs)")
    print("=" * 80)
    pairs = {k: v for k, v in freq.items() if len(k) == 2}
    top_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:15]
    for itemset, count in top_pairs:
        items = sorted(itemset)
        print(f"  {items[0]} + {items[1]}: {count} transactions ({count/n*100:.2f}%)")

    # ─── Business Recommendations ──────────────────────────
    print("\n" + "=" * 80)
    print("  BUSINESS RECOMMENDATIONS")
    print("=" * 80)

    # High-lift rules suggest strong associations
    high_lift = [r for r in rules_sorted if r['lift'] > 2.0]
    print(f"\n  {len(high_lift)} rules with lift > 2.0 (strong positive association):")
    for r in high_lift[:10]:
        print(f"    {format_rule(r)} (lift={r['lift']:.2f})")

    print("""
  Key Insights & Recommendations:
  ─────────────────────────────────
  1. PRODUCT PLACEMENT: Items frequently bought together should be
     placed near each other in the store to encourage cross-buying.

  2. BUNDLE PROMOTIONS: Create discount bundles for high-confidence
     pairs to increase basket size.

  3. RECOMMENDATION ENGINE: Use high-lift rules to power a
     "Customers also bought..." feature online.

  4. INVENTORY MANAGEMENT: Stock associated items together to
     avoid one being out of stock when the other is available.

  5. TARGETED COUPONS: If a customer buys item A, send them a
     coupon for item B (based on high-confidence rules).
""")
