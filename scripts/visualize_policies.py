#!/usr/bin/env python3
"""
Compares the ~19 distinct ODRL policy profiles produced across the corpus:
  1. A Jaccard similarity heatmap between license profiles (how much of
     their permission/prohibition/duty rule set overlaps), ordered by
     hierarchical clustering so similar licenses land next to each other.
  2. A presence/absence matrix: every license x every (rule type, action)
     pair that appears anywhere in the corpus, showing exactly which
     rules are shared and which are unique to one license.

Both figures encode file count (how many real datasets use each license)
alongside the structural comparison, via a marginal bar chart and label
annotations — not just abstract policy shape, but how much real-world
weight each shape actually carries.

Usage:
    python3 visualize_policies.py <path-to-hf-corpus-10k>
"""
import sys
import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def load_corpus(corpus_dir):
    """Groups files by identical usageInfo, returns {name: (usageInfo, count, sample_file)}."""
    groups = defaultdict(list)
    for f in glob.glob(f"{corpus_dir}/*.jsonld"):
        d = json.load(open(f))
        ui = d.get("usageInfo")
        if ui is None:
            continue
        key = json.dumps(ui, sort_keys=True)
        groups[key].append(f)

    licenses = {}
    for key, files in groups.items():
        ui = json.loads(key)
        name = ui.get("name", "?")
        licenses[name] = (ui, len(files))
    return licenses


def rule_actions(ui, rule_key):
    """Extracts the set of action strings for a rule type (permission/prohibition/duty).
    Handles both single-dict and list-of-dicts shapes — inject_odrl stores a
    single rule as a bare object, multiple as a list."""
    raw = ui.get(rule_key)
    if raw is None:
        return set()
    rules = raw if isinstance(raw, list) else [raw]
    actions = set()
    for r in rules:
        action = r.get("odrl:action", {})
        aid = action.get("@id", "") if isinstance(action, dict) else str(action)
        actions.add(aid.replace("odrl:", ""))
    return actions


def profile_signature(ui):
    """(ruleType, action) pairs — the full comparable fingerprint of a policy."""
    sig = set()
    for rule_key, label in [("odrl:permission", "permit"), ("odrl:prohibition", "prohibit"), ("odrl:duty", "duty")]:
        for action in rule_actions(ui, rule_key):
            sig.add(f"{label}:{action}")
    return sig


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_policies.py <path-to-hf-corpus-10k>")
        sys.exit(1)

    licenses = load_corpus(sys.argv[1])
    names = list(licenses.keys())
    signatures = {n: profile_signature(licenses[n][0]) for n in names}
    counts = {n: licenses[n][1] for n in names}

    n = len(names)
    print(f"{n} distinct license profiles, {sum(counts.values())} files total")

    # --- Similarity matrix + clustering order ---
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = jaccard(signatures[names[i]], signatures[names[j]])

    if HAVE_SCIPY and n > 2:
        dist = 1 - sim
        np.fill_diagonal(dist, 0)
        condensed = dist[np.triu_indices(n, k=1)]
        Z = linkage(condensed, method="average")
        order = dendrogram(Z, no_plot=True)["leaves"]
    else:
        order = sorted(range(n), key=lambda i: -counts[names[i]])

    ordered_names = [names[i] for i in order]
    ordered_sim = sim[np.ix_(order, order)]
    labels = [f"{nm}  (n={counts[nm]})" for nm in ordered_names]

    # --- Figure 1: similarity heatmap + count marginal ---
    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 9), gridspec_kw={"width_ratios": [4, 1]}, constrained_layout=True
    )
    im = ax_heat.imshow(ordered_sim, cmap="viridis", vmin=0, vmax=1)
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(ordered_names, rotation=90, fontsize=8)
    ax_heat.set_yticklabels(labels, fontsize=8)
    ax_heat.set_title("Policy similarity (Jaccard over permission/prohibition/duty actions)")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="Jaccard similarity")

    bar_counts = [counts[nm] for nm in ordered_names]
    ax_bar.barh(range(n), bar_counts, color="steelblue")
    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_xscale("log")
    ax_bar.set_xlabel("files (log scale)")
    ax_bar.set_title("File count")
    ax_heat.invert_yaxis()
    ax_bar.set_ylim(ax_heat.get_ylim())

    fig.savefig("policy_similarity_heatmap.png", dpi=150)
    print("Wrote policy_similarity_heatmap.png")

    # --- Figure 2: presence/absence matrix over every (rule, action) pair ---
    all_terms = sorted(set().union(*signatures.values()))
    presence = np.zeros((n, len(all_terms)))
    for i, nm in enumerate(ordered_names):
        for j, term in enumerate(all_terms):
            presence[i, j] = 1 if term in signatures[nm] else 0

    fig2, ax2 = plt.subplots(figsize=(max(8, len(all_terms) * 0.4), 9), constrained_layout=True)
    ax2.imshow(presence, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(all_terms)))
    ax2.set_xticklabels(all_terms, rotation=90, fontsize=8)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_title("Which rules each license profile actually contains")
    fig2.savefig("policy_presence_matrix.png", dpi=150)
    print("Wrote policy_presence_matrix.png")


if __name__ == "__main__":
    main()
