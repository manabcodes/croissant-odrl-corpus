# croissant-odrl-corpus

4,078 Croissant 1.1 metadata files for real Hugging Face datasets, each with an
ODRL usage policy (`usageInfo` / `odrl:Offer`) derived from the dataset's declared
license. Produced during a COST GOBLIN (CA23147) short-term scientific mission
at IDLab, Ghent University.

- `corpus/`: the 4,078 `.jsonld` files and `manifest.csv`
- `scripts/`: generation script (`bulk_hf_odrl_corpus_2.py`), policy config, run log, visualisation
- `patch/`: patch adding ODRL authoring (`--odrl-config`, `odrl-apply`) to croissant-baker
- `figures/`: policy presence matrix and similarity heatmap

Policies come from 19 license profiles applied to declared license tags, not from
dataset owners, so they validate tooling and do not state publisher intent.
Datasets without a mapped license were skipped (see `manifest.csv`).
The evaluator is in `croissant-odrl-evaluator`.
