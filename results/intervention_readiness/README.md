# Intervention-readiness extension: current results

## What is complete

- Four known CRC mechanisms were retained as a frozen benchmark.
- A country-aware, cross-fitted screen evaluated 6,755 UniRef90 gene families.
- Sixteen gene families met the frozen internal nomination rules.
- UniProt/UniParc provenance was resolved for every nomination.
- A provisional atlas separates known **effector targets** from possible
  strain-selective **precision addresses**.
- One explicit-gate evidence dossier was generated for every atlas entry.

## Current scientific signal

The known mechanisms recur directionally but are weak abundance classifiers:
colibactin is CRC-enriched in 9 of 10 evaluable cohorts with median association
AUC 0.541, while no benchmark passes the frozen cross-population gate. This is
not used to discard the mechanisms or infer that they are biologically
unimportant.

The discovery track produced 16 internally cross-fitted nominations. The
strongest, `UniRef90_A0A0E2AL27`, was selected in all ten outer folds, was
CRC-enriched in 80% of corresponding held-out cohorts, and had median held-out
AUC 0.635. Automated annotation resolves it to a short uncharacterized protein
from an archived *Bacteroides fragilis* genome. That makes it a possible
strain-address lead—not a causal gene or CRISPR guide.

Ten of the sixteen nominations are uncharacterized. The annotated remainder
include ordinary enzymes, regulators, recombinases, and structural machinery.
The next analyses must determine whether these are portable strain addresses,
taxonomic passengers, or technical aliases.

## What remains before a publishable intervention claim

1. Finish taxon-resolved attribution and parent-strain adjustment.
2. Freeze required-gene rules for known effector mechanisms.
3. Test sequence conservation within intended harmful strains.
4. Screen specificity against protected commensals and human sequence.
5. Complete the structured causality and delivery/editability evidence review.
6. Confirm the nominated sequences in an untouched external cohort at gene
   level. Existing external species profiles do not satisfy this requirement.
7. Obtain Rachel's biological review before freezing the final shortlist.

Until those gates are complete, `readiness_atlas.csv` is a research-priority
atlas, not an experiment-ready or treatment-ready list.
