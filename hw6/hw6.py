# Zongyang Li
# 02/28/2026

import networkx as nx
import matplotlib.pyplot as plt
import re
import string
from collections import Counter


# Step 1: Read the text file
def read_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


# Step 2: Download stop words list
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))


# Step 3: Tokenize raw text into sentences -> words
def tokenize_raw(text):
    """
    Tokenize the raw text. Split into sentences, then split each sentence into words.
    Keep original casing and punctuation attached to words (raw).
    """
    # Split text into sentences by common sentence delimiters
    sentences = re.split(r'[.!?]+', text)
    all_sentence_words = []
    for sent in sentences:
        # Split sentence into words by whitespace
        words = sent.split()
        # Remove empty strings
        words = [w for w in words if w.strip()]
        if words:
            all_sentence_words.append(words)
    return all_sentence_words


# Step 4: Tokenize cleaned text into sentences -> words
def tokenize_cleaned(text):
    """
    Clean the text:
    1. Lowercase everything
    2. Remove punctuation
    3. Remove stop words
    Then tokenize into sentences -> words.
    """
    # Split text into sentences first (before removing punctuation)
    sentences = re.split(r'[.!?]+', text)
    all_sentence_words = []
    for sent in sentences:
        # Lowercase
        sent = sent.lower()
        # Remove punctuation
        sent = sent.translate(str.maketrans('', '', string.punctuation))
        # Remove special characters (em dash, etc.)
        sent = re.sub(r'[—–\u2014\u2013]', ' ', sent)
        # Split into words
        words = sent.split()
        # Remove stop words
        words = [w for w in words if w not in STOP_WORDS and w.strip()]
        if words:
            all_sentence_words.append(words)
    return all_sentence_words


# Step 5: Build word network graph
def build_word_network(sentence_words_list):
    """
    Build a NetworkX graph where:
    - Each unique word is a node
    - Edges connect sequential words within each sentence
    - Node size is proportional to word frequency
    """
    G = nx.DiGraph()
    word_freq = Counter()

    for words in sentence_words_list:
        # Count word frequencies
        for w in words:
            word_freq[w] += 1
        # Add edges between sequential words
        for i in range(len(words) - 1):
            w1 = words[i]
            w2 = words[i + 1]
            if G.has_edge(w1, w2):
                G[w1][w2]['weight'] += 1
            else:
                G.add_edge(w1, w2, weight=1)

    return G, word_freq


# Step 6: Draw the word network graph
def draw_word_network(G, word_freq, title, filename):
    """
    Draw the word network with node sizes proportional to word frequency.
    """
    plt.figure(figsize=(16, 12))

    # Use spring layout for nice visualization
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

    # Calculate node sizes based on frequency (scale for visibility)
    node_sizes = [word_freq.get(node, 1) * 300 for node in G.nodes()]

    # Calculate edge widths based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + (w / max_weight) * 3 for w in edge_weights]

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.3,
        edge_color='gray',
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1"
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='lightblue',
        edgecolors='darkblue',
        linewidths=1.5,
        alpha=0.8
    )

    # Draw labels
    font_sizes = {node: max(7, min(14, word_freq.get(node, 1) * 2)) for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight='bold'
    )

    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# Step 7: Print graph statistics
def print_stats(G, word_freq, label):
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Number of nodes (unique words): {G.number_of_nodes()}")
    print(f"  Number of edges (word connections): {G.number_of_edges()}")

    # Top 10 most frequent words
    top_words = word_freq.most_common(10)
    print(f"\n  Top 10 most frequent words:")
    for word, freq in top_words:
        print(f"    '{word}': {freq}")

    # Top 10 nodes by degree (most connected)
    degree_dict = dict(G.degree())
    top_degree = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top 10 most connected words (by degree):")
    for word, deg in top_degree:
        print(f"    '{word}': {deg} connections")


# Main Execution
if __name__ == "__main__":
    filepath = "data/BeachDog.txt"
    text = read_text(filepath)

    # Show text cleaning process

    print("=" * 60)
    print("  TEXT CLEANING DEMONSTRATION")
    print("=" * 60)

    # Show original raw text
    print("\n--- ORIGINAL RAW TEXT ---")
    print(text)

    # Show cleaned text
    print("\n--- CLEANED TEXT ---")
    print("(Lowercased, punctuation removed, stop words removed)\n")
    cleaned_sentences = tokenize_cleaned(text)
    for i, sentence_words in enumerate(cleaned_sentences):
        print(f"  Sentence {i + 1}: {' '.join(sentence_words)}")

    # Show what was removed
    raw_sentences = tokenize_raw(text)
    raw_words = [w for sent in raw_sentences for w in sent]
    cleaned_words = [w for sent in cleaned_sentences for w in sent]

    print(f"\n--- CLEANING SUMMARY ---")
    print(f"  Total words before cleaning: {len(raw_words)}")
    print(f"  Total words after cleaning:  {len(cleaned_words)}")
    print(f"  Words removed:               {len(raw_words) - len(cleaned_words)}")
    print(f"  Removal rate:                {(len(raw_words) - len(cleaned_words)) / len(raw_words) * 100:.1f}%")

    # Show which stop words were removed
    raw_lower = [w.lower().strip(string.punctuation) for w in raw_words]
    removed_stops = [w for w in raw_lower if w in STOP_WORDS and w != '']
    stop_freq = Counter(removed_stops)
    print(f"\n  Stop words removed (with counts):")
    for word, count in stop_freq.most_common():
        print(f"    '{word}': {count}")

    # Build and draw networks

    # Raw Text Network
    G_raw, freq_raw = build_word_network(raw_sentences)
    print_stats(G_raw, freq_raw, "RAW TEXT WORD NETWORK")
    draw_word_network(G_raw, freq_raw,
                      "Word Network - Raw Text (BeachDog.txt)",
                      "hw6_output/raw_word_network.png")

    # Cleaned Text Network
    G_clean, freq_clean = build_word_network(cleaned_sentences)
    print_stats(G_clean, freq_clean, "CLEANED TEXT WORD NETWORK")
    draw_word_network(G_clean, freq_clean,
                      "Word Network - Cleaned Text (BeachDog.txt)",
                      "hw6_output/cleaned_word_network.png")

    print("\n\nDone! Two graph images have been saved.")
    print(f"  Raw network:     {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges")
    print(f"  Cleaned network: {G_clean.number_of_nodes()} nodes, {G_clean.number_of_edges()} edges")
    print(f"  Nodes removed by cleaning: {G_raw.number_of_nodes() - G_clean.number_of_nodes()}")