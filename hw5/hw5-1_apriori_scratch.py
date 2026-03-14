"""
Part 1: Apriori Algorithm Implementation from Scratch
"""

import csv
from itertools import combinations
from collections import defaultdict


# 1. Load Data

def load_transactions(filepath):
    """Load CSV where each row is a transaction (variable-length items)."""
    transactions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # Strip whitespace and filter out empty strings
            items = [item.strip() for item in row if item.strip()]
            if items:
                transactions.append(set(items))
    print(f"Loaded {len(transactions)} transactions.")
    return transactions



# 2. Core Apriori Functions

def get_support(itemset, transactions):
    """Calculate support count of an itemset."""
    count = 0
    for t in transactions:
        if itemset.issubset(t):
            count += 1
    return count


def get_L1(transactions, min_support_count):
    """Generate frequent 1-itemsets (L1)."""
    item_counts = defaultdict(int)
    for t in transactions:
        for item in t:
            item_counts[frozenset([item])] += 1

    L1 = {itemset: count for itemset, count in item_counts.items()
           if count >= min_support_count}
    return L1


def has_infrequent_subset(candidate, Lk_minus_1):
    """Pruning step: check if any (k-1)-subset of candidate is NOT in L_{k-1}."""
    k = len(candidate)
    for subset in combinations(candidate, k - 1):
        if frozenset(subset) not in Lk_minus_1:
            return True
    return False


def apriori_gen(Lk_minus_1):
    """
    Candidate generation: join step + prune step.
    Lk_minus_1: dict of frequent (k-1)-itemsets -> support count
    Returns candidate k-itemsets.
    """
    candidates = set()
    itemsets = list(Lk_minus_1.keys())

    for i in range(len(itemsets)):
        for j in range(i + 1, len(itemsets)):
            # Join: union two (k-1)-itemsets that share k-2 items
            union = itemsets[i] | itemsets[j]
            if len(union) == len(itemsets[i]) + 1:
                # Prune: check all (k-1)-subsets are frequent
                if not has_infrequent_subset(union, Lk_minus_1):
                    candidates.add(union)
    return candidates


def apriori(transactions, min_support=0.01):
    """
    Full Apriori algorithm.
    Returns all frequent itemsets with their support counts.
    """
    n = len(transactions)
    min_support_count = int(min_support * n)
    print(f"Min support = {min_support} => min support count = {min_support_count}")
    print("=" * 70)

    # Step 1: L1
    L1 = get_L1(transactions, min_support_count)
    print(f"\n--- L1: {len(L1)} frequent 1-itemsets ---")
    # Print top 20 by support
    sorted_L1 = sorted(L1.items(), key=lambda x: x[1], reverse=True)
    for itemset, count in sorted_L1[:20]:
        items_str = ", ".join(sorted(itemset))
        print(f"  {{{items_str}}}: support = {count}/{n} = {count/n:.4f}")
    if len(sorted_L1) > 20:
        print(f"  ... and {len(sorted_L1) - 20} more items")

    all_frequent = dict(L1)
    k = 2
    Lk_prev = L1

    # Step 2: Iterate
    while Lk_prev:
        # Generate candidates
        candidates = apriori_gen(Lk_prev)
        if not candidates:
            break

        # Count support for each candidate
        candidate_counts = defaultdict(int)
        for t in transactions:
            for c in candidates:
                if c.issubset(t):
                    candidate_counts[c] += 1

        # Filter by min support
        Lk = {itemset: count for itemset, count in candidate_counts.items()
               if count >= min_support_count}

        if not Lk:
            print(f"\n--- L{k}: 0 frequent {k}-itemsets (stopping) ---")
            break

        print(f"\n--- L{k}: {len(Lk)} frequent {k}-itemsets ---")
        sorted_Lk = sorted(Lk.items(), key=lambda x: x[1], reverse=True)
        display_count = min(20, len(sorted_Lk))
        for itemset, count in sorted_Lk[:display_count]:
            items_str = ", ".join(sorted(itemset))
            print(f"  {{{items_str}}}: support = {count}/{n} = {count/n:.4f}")
        if len(sorted_Lk) > display_count:
            print(f"  ... and {len(sorted_Lk) - display_count} more itemsets")

        all_frequent.update(Lk)
        Lk_prev = Lk
        k += 1

    print(f"\nTotal frequent itemsets found: {len(all_frequent)}")
    return all_frequent, n



# 3. Association Rule Generation

def generate_rules(frequent_itemsets, n, min_confidence=0.3):
    """
    Generate association rules from frequent itemsets.
    For each itemset of size >= 2, generate all non-empty proper subsets
    as antecedents.
    """
    rules = []
    for itemset, support_count in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        support = support_count / n

        # Generate all non-empty proper subsets as antecedent
        items = list(itemset)
        for i in range(1, len(items)):
            for antecedent in combinations(items, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent

                # Confidence = support(A ∪ B) / support(A)
                ant_support_count = frequent_itemsets.get(antecedent, 0)
                if ant_support_count == 0:
                    continue
                confidence = support_count / ant_support_count

                if confidence >= min_confidence:
                    # Lift = confidence / support(B)
                    cons_support_count = frequent_itemsets.get(consequent, 0)
                    if cons_support_count == 0:
                        continue
                    lift = confidence / (cons_support_count / n)

                    rules.append({
                        'antecedent': antecedent,
                        'consequent': consequent,
                        'support': support,
                        'confidence': confidence,
                        'lift': lift
                    })
    return rules


def print_rules(rules, top_n=30):
    """Print association rules sorted by lift."""
    rules_sorted = sorted(rules, key=lambda x: x['lift'], reverse=True)
    print(f"\n{'='*90}")
    print(f"Top {min(top_n, len(rules_sorted))} Association Rules (sorted by lift)")
    print(f"{'='*90}")
    print(f"{'Rule':<55} {'Support':>8} {'Confidence':>11} {'Lift':>8}")
    print("-" * 90)

    for r in rules_sorted[:top_n]:
        ant = ", ".join(sorted(r['antecedent']))
        con = ", ".join(sorted(r['consequent']))
        rule_str = f"{{{ant}}} => {{{con}}}"
        print(f"{rule_str:<55} {r['support']:>8.4f} {r['confidence']:>11.4f} {r['lift']:>8.4f}")

    return rules_sorted



# 4. Main

if __name__ == "__main__":
    filepath = "data/Market_Basket_Optimisation.csv"

    print("=" * 70)
    print("  PART 1: Apriori Algorithm - Implementation from Scratch")
    print("=" * 70)

    transactions = load_transactions(filepath)

    # Run Apriori with min_support = 0.01 (1%)
    MIN_SUPPORT = 0.01
    MIN_CONFIDENCE = 0.3

    print(f"\nRunning Apriori with min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE}")
    frequent_itemsets, n = apriori(transactions, min_support=MIN_SUPPORT)

    # Generate and print rules
    rules = generate_rules(frequent_itemsets, n, min_confidence=MIN_CONFIDENCE)
    print(f"\nGenerated {len(rules)} association rules with confidence >= {MIN_CONFIDENCE}")
    rules_sorted = print_rules(rules, top_n=30)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total transactions: {n}")
    print(f"Total frequent itemsets: {len(frequent_itemsets)}")
    print(f"Total rules (confidence >= {MIN_CONFIDENCE}): {len(rules)}")

    # Top 5 most frequent items
    single_items = {k: v for k, v in frequent_itemsets.items() if len(k) == 1}
    top_items = sorted(single_items.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 most purchased items:")
    for itemset, count in top_items:
        print(f"  {list(itemset)[0]}: {count} ({count/n*100:.1f}%)")

    # Top 5 most frequent pairs
    pairs = {k: v for k, v in frequent_itemsets.items() if len(k) == 2}
    top_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 most frequent item pairs:")
    for itemset, count in top_pairs:
        print(f"  {set(itemset)}: {count} ({count/n*100:.1f}%)")
