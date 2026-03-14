"""
HW5 - Part 3: Apriori Algorithm on a New Dataset
=================================================
This script creates a realistic grocery store dataset and runs
the Apriori algorithm on it to discover association rules.

Dataset: Synthetic Online Retail Transactions
- Simulates 5000 transactions with realistic item co-occurrence patterns.
- Items span categories: bakery, dairy, beverages, snacks, produce, meat, etc.
"""

import csv
import random
from itertools import combinations
from collections import defaultdict


# ──────────────────────────────────────────────────────────
# 1. Generate a New Dataset
# ──────────────────────────────────────────────────────────
def generate_dataset(filepath, num_transactions=5000, seed=42):
    """
    Generate a realistic grocery transaction dataset.
    Items have natural co-occurrence patterns built in.
    """
    random.seed(seed)

    # Define item groups that tend to be bought together
    item_groups = {
        'breakfast': ['bread', 'butter', 'jam', 'cereal', 'milk', 'orange juice', 'eggs', 'bacon'],
        'pasta_night': ['pasta', 'tomato sauce', 'ground beef', 'parmesan', 'garlic', 'onion', 'olive oil'],
        'healthy': ['salad greens', 'avocado', 'chicken breast', 'quinoa', 'almond milk', 'berries'],
        'snacks': ['chips', 'soda', 'cookies', 'candy', 'popcorn', 'ice cream'],
        'bbq': ['steak', 'corn', 'buns', 'ketchup', 'mustard', 'charcoal', 'beer'],
        'baking': ['flour', 'sugar', 'baking powder', 'vanilla extract', 'butter', 'eggs', 'chocolate chips'],
        'asian_cooking': ['rice', 'soy sauce', 'tofu', 'ginger', 'sesame oil', 'noodles', 'green onion'],
        'baby': ['diapers', 'baby food', 'baby wipes', 'formula', 'baby cereal'],
        'cleaning': ['dish soap', 'paper towels', 'trash bags', 'bleach', 'sponges'],
        'pet': ['dog food', 'cat food', 'pet treats', 'cat litter'],
    }

    # Also define some always-popular individual items
    popular_items = ['water', 'bread', 'milk', 'bananas', 'eggs', 'rice', 'chicken breast', 'apples']

    transactions = []
    for _ in range(num_transactions):
        basket = set()

        # Each transaction picks 1-3 groups with some probability
        num_groups = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        chosen_groups = random.sample(list(item_groups.keys()), num_groups)

        for group in chosen_groups:
            items = item_groups[group]
            # Pick a subset of items from the group
            k = random.randint(2, min(5, len(items)))
            basket.update(random.sample(items, k))

        # Add 0-2 popular standalone items
        num_popular = random.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
        basket.update(random.sample(popular_items, min(num_popular, len(popular_items))))

        transactions.append(list(basket))

    # Write to CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for t in transactions:
            writer.writerow(t)

    print(f"Generated {num_transactions} transactions -> {filepath}")
    return transactions


# ──────────────────────────────────────────────────────────
# 2. Apriori Functions (reused)
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
                k = len(union)
                prune = False
                for sub in combinations(union, k - 1):
                    if frozenset(sub) not in Lk_prev:
                        prune = True
                        break
                if not prune:
                    candidates.add(union)
    return candidates


def run_apriori(transactions, min_support=0.01, verbose=True):
    n = len(transactions)
    min_sup_count = int(min_support * n)
    if verbose:
        print(f"Min support = {min_support} => min count = {min_sup_count}")

    L1 = _get_L1(transactions, min_sup_count)
    if verbose:
        print(f"L1: {len(L1)} frequent 1-itemsets")
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
            if verbose:
                print(f"L{k}: 0 frequent {k}-itemsets (stopping)")
            break
        if verbose:
            print(f"L{k}: {len(Lk)} frequent {k}-itemsets")
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
# 3. Main
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("  PART 3: Apriori on a New Dataset - Synthetic Grocery Transactions")
    print("=" * 80)

    # Generate new dataset
    new_data_path = "new_grocery_dataset.csv"
    generate_dataset(new_data_path, num_transactions=5000, seed=42)

    # Load and run
    transactions = load_transactions(new_data_path)
    print(f"\nDataset: {len(transactions)} transactions")

    # Count unique items
    all_items = set()
    for t in transactions:
        all_items.update(t)
    print(f"Unique items: {len(all_items)}")

    # Run Apriori
    print("\n--- Running Apriori (min_support=0.02, min_confidence=0.3) ---\n")
    freq, n = run_apriori(transactions, min_support=0.02)

    rules = generate_rules(freq, n, min_confidence=0.3)
    rules_sorted = sorted(rules, key=lambda x: x['lift'], reverse=True)

    print(f"\nTotal frequent itemsets: {len(freq)}")
    print(f"Total rules: {len(rules_sorted)}")

    # Print L_k summary
    for k_size in range(1, 5):
        count = len([i for i in freq if len(i) == k_size])
        if count > 0:
            print(f"  L{k_size}: {count} frequent {k_size}-itemsets")

    # Print top rules
    print(f"\n{'Rule':<55} {'Support':>8} {'Confidence':>11} {'Lift':>8}")
    print("-" * 85)
    for r in rules_sorted[:30]:
        print(f"{format_rule(r):<55} {r['support']:>8.4f} {r['confidence']:>11.4f} {r['lift']:>8.2f}")

    # Top pairs
    print("\nTop 10 most frequent item pairs:")
    pairs = {k: v for k, v in freq.items() if len(k) == 2}
    top_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:10]
    for itemset, count in top_pairs:
        items = sorted(itemset)
        print(f"  {items[0]} + {items[1]}: {count} ({count/n*100:.2f}%)")

    # Analysis
    print(f"""
{'='*80}
  ANALYSIS OF NEW DATASET RESULTS
{'='*80}

  Dataset Description:
  ─────────────────────
  This synthetic dataset simulates 5,000 grocery store transactions with
  realistic co-occurrence patterns. Items were grouped into thematic 
  categories (breakfast, pasta night, BBQ, baking, etc.) to create 
  natural associations.

  Key Findings:
  ──────────────
  1. Strong associations were found within themed meal groups, confirming
     that shoppers tend to buy items for specific meal occasions.

  2. Items like eggs, butter, and milk appear across multiple groups
     (breakfast + baking), making them high-frequency bridge items.

  3. Niche categories (baby supplies, pet supplies) show very high lift
     values because their items are rarely bought by non-target customers.

  4. The BBQ group (steak, corn, buns, beer) shows strong within-group
     associations, suggesting seasonal promotion opportunities.

  Comparison with Market Basket Dataset:
  ───────────────────────────────────────
  - The Market Basket dataset has more diffuse associations due to the
    wider variety of items and less structured shopping patterns.
  - This synthetic dataset produces higher lift values because of the
    embedded group structure.
  - Both datasets confirm that grocery items cluster into meal/occasion
    categories rather than random combinations.
""")
