# Readout card

**Source evidence:** `artifacts/dga-domain-detector/src/pages/home.tsx`,
`ResultPanel`, `ModelSummary`, and `HistoryPanel`.

Cards are grouped with a soft 12px radius, quiet border, warm surface, and
restrained shadow. The verdict header uses a tinted semantic band; the body
keeps decision score and feature values prominent while supporting copy stays
muted. Dense metadata can sit in a right rail on large screens and stack below
on narrow screens.

**Design-system mapping:** use Card and muted text primitives, preserving
space for product-specific compositions.