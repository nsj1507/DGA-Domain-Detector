# Domain input

**Source evidence:** `artifacts/dga-domain-detector/src/pages/home.tsx`,
`domainSchema`, and the input channel.

The domain field is a full-width 56px control with monospaced text, a warm
canvas fill, a quiet border, and a lime focus ring. Validation is shown below
the field as a concise inline attention message. The domain is submitted as a
string only; DNS or external enrichment is not implied by the surface.

**Design-system mapping:** use the Input family with a mono type token,
comfortable horizontal padding, and error text outside the control.