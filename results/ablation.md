# Ablation Study

Mean RobF1 by problem group (10 runs per problem, same seeds and codebase as the main
experiment). Relative change vs. full TSRDE in parentheses.

| Variant | Grp-A | Grp-B | Grp-C | Grp-D | Grp-E | Overall |
|---|---|---|---|---|---|---|
| TSRDE (full) | 0.887 | 1.000 | 0.987 | 0.886 | 1.000 | **0.952** |
| Abl-MMO | 0.775 | 0.988 | 0.963 | 0.671 | 1.000 | 0.879 (-7.6%) |
| Abl-noRestart | 0.854 | 1.000 | 0.977 | 0.860 | 1.000 | 0.938 (-1.4%) |
| Abl-noElim | 0.621 | 0.721 | 0.759 | 0.713 | 0.721 | 0.707 (-25.8%) |