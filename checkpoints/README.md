# Pretrained FateDriver weights

Pretrained weights (seed 42) are ~113 MB per dataset and are therefore not
tracked in git. They are **not** part of the Zenodo data record
([10.5281/zenodo.21771206](https://doi.org/10.5281/zenodo.21771206)), which
holds the datasets and gene sets only — request the checkpoints from the
authors, or reproduce them with `bash scripts/run_fatedriver.sh <dataset> 42`.

Each dataset's weights live here:

```
checkpoints/<dataset>/
├── cell_encoder.pt
├── gene_feature_projector.pt
├── gene_encoder.pt
├── set_encoder.pt
├── ranking_head.pt
├── driver_prior_model.pt
├── driver_prior_metadata.json     # learned prior feature weights
├── recon_params.pt                # gene bias, library scale, ZINB dispersion/dropout
├── predictions.csv                # gene,score ranking produced by this checkpoint
└── meta.json                      # full run configuration
```

`<dataset>` is one of: `GSE242423`, `GSE297234`, `GSE98567`,
`GSE181451_Primed_MEF`, `GSE181451_Baseline_MEF`. Note that the `GSE297234`
checkpoint was trained on both donors merged, while the released data (and
`data/datasets.tsv`) provide the two donors separately as `GSE297234_GM00731`
and `GSE297234_GM23815`.

These are the exact weights behind the paper's reported numbers. Re-running
`scripts/run_fatedriver.sh <dataset> 42` from the released data reproduces them
up to GPU non-determinism. The authoritative configuration for each checkpoint
is its own `meta.json`.
