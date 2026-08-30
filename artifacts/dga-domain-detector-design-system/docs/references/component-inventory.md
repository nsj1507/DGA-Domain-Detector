# DGA Domain Detector component inventory

This inventory is extracted from the existing DGA Domain Detector web artifact.
The source is an application surface rather than a separately published
component library, so product-specific compositions remain documented as
patterns while the design-system package provides the reusable primitives.

| Family | Reference | Dependencies/blockers | Evidence | Status |
| --- | --- | --- | --- | --- |
| Analyst action | `components/analyst-action.md` | Button primitive, icon slot | Run analysis and reset actions in the input channel | implemented |
| Domain input | `components/domain-input.md` | Input primitive, form validation | Monospaced domain field with inline search icon | implemented |
| Readout card | `components/readout-card.md` | Card primitive, border/elevation tokens | Verdict, model summary, and session history surfaces | implemented |
| Risk badge | `components/risk-badge.md` | Badge primitive, semantic colors | Low/high risk and live system labels | implemented |
| System alert | `components/system-alert.md` | Alert primitive, semantic colors | Model unavailable and analysis error states | implemented |
| Disclosure and loading | `components/disclosure-loading.md` | Collapsible and Skeleton primitives | Model configuration disclosure and loading blocks | implemented |

The source does not ship a standalone logo asset. The header mark is an
application-level Radar icon, so no logo is retained or invented in this
package.