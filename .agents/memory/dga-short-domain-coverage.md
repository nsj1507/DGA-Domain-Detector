---
name: DGA short-domain coverage
description: Training-data and validation constraints for short DGA-like domains
---

The source DGA-positive data has a strong length gap: its registrable labels begin at seven characters, while the legitimate class contains many one-to-six-character labels. A high overall score can therefore hide short-DGA false negatives.

**Why:** The model can appear correct on the original holdout while treating short, random-looking domains as legitimate because that class boundary was absent from positive training data.

**How to apply:** Keep generated short-DGA coverage and a fixed challenge matrix in model validation. Treat the persisted `0 = Legitimate`, `1 = Malicious` label contract and the 19-handcrafted-then-10,000-TF-IDF feature order as invariants.