---
name: DGA short-domain coverage
description: Training-data and validation constraints for short DGA-like domains
---

The original DGA-positive data had a strong length gap: its registrable labels began at seven characters, while the legitimate class contained many one-to-six-character labels. Training now supplements it with deduplicated, source-derived short DGA rows from the verified ExtraHop corpus rather than generated random strings.

**Why:** The model can appear correct on the original holdout while treating short, random-looking domains as legitimate because that class boundary was absent from positive training data.

**How to apply:** Keep verified short-DGA coverage, leakage checks, and a fixed challenge matrix in model validation. Treat the persisted `0 = Legitimate`, `1 = Malicious` label contract and the 19-handcrafted-then-10,000-TF-IDF feature order as invariants.