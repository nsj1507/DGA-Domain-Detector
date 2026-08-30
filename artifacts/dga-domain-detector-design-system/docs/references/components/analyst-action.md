# Analyst action

**Source evidence:** `artifacts/dga-domain-detector/src/pages/home.tsx`,
input channel action row.

Actions use a deep teal filled primary action for the main workflow and a
quiet outlined reset action. The primary action includes a small leading and
trailing icon, lifts by one pixel on hover, and becomes visibly disabled while
the detector is processing or unavailable.

**Design-system mapping:** use the Button family with primary and outline
variants, compact 44px action height, and the `accent` color only for signal
decoration rather than the main action fill.