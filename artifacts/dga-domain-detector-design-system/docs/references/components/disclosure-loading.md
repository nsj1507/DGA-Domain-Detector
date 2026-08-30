# Disclosure and loading

**Source evidence:** `artifacts/dga-domain-detector/src/pages/home.tsx`,
model configuration disclosure and model summary loading state.

Secondary model metadata is progressively disclosed behind a text trigger with
a chevron. Loading states preserve the final layout using a small set of
animated neutral blocks rather than a full-screen spinner.

**Design-system mapping:** use Collapsible for configuration sections and
Skeleton for reserved panel content. Respect reduced-motion preferences.