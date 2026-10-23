# FateDriver

**FateDriver** identifies the driver genes of a cell-state transition (e.g.
somatic cell → iPSC reprogramming) from single-cell RNA-seq. It places cells and
genes in a shared latent space and defines a transition as the direction from a
pooled **source** state to a pooled **target** state, scoring every gene by how
closely it aligns with that direction.

Only the delivered factors and their annotated downstream targets are labeled;
the remaining genes are untested regulators rather than confirmed non-drivers.
FateDriver therefore **ranks** genes instead of classifying them — a
positive-unlabeled problem (Bekker & Davis, 2020). The unlabeled genes that
align with the transition as the delivered factors do are the candidate
regulators it reports. A score consequently belongs to a **specified
transition**, not to a dataset.

This repository contains the source code, environment specification, released
datasets and pretrained weights needed to reproduce the FateDriver results.

Pipeline at a glance:

```
preprocessed AnnData (.h5ad)
        │  scripts/compute_pseudotime.py      (§2.1  slingshot)
        ▼
   pseudotime.csv
        │  scripts/prepare_dataset.py         (§2.2)
        ▼
data/<dataset>/{anndata.h5ad, pseudotime.csv} (§3  input format)
        │  src/train_fatedriver.py            (§4)
        ▼
   predictions.csv  (ranked driver genes)     (§5)
```

---

## 1. Environment setup

FateDriver needs Python 3.10 and a CUDA 12.1 capable GPU (training is only
practical on GPU; inference from saved weights also runs on CPU).

```bash
# from the repository root
conda env create --solver=libmamba -f environment.yml
conda activate FateDriver
```

`--solver=libmamba` is recommended: with the classic solver (the default before
conda 23.10) resolving the conda-forge channel can take a very long time. On
conda ≥ 23.10 libmamba is already the default and the flag can be dropped; on
older conda install it first with `conda install -n base conda-libmamba-solver`.

This installs PyTorch 2.5.1 (CUDA 12.1) plus the pinned scientific stack
(numpy, scipy, pandas, scikit-learn, anndata, h5py, umap-learn, matplotlib,
scanpy). On a CPU-only machine, edit `environment.yml` (see the comment at the
top).

Verify the install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# -> 2.5.1+cu121 True
```

**Optional — R environment for pseudotime.** The released `pseudotime.csv`
files were produced with Slingshot. Only needed if you want to recompute
pseudotime for your own data with the same method (§2.1):

```bash
conda env create --solver=libmamba -f environment_pseudotime.yml   # r-base 4.5 + slingshot 2.18.0
```

You do **not** need it to run FateDriver on the released data, or if you use
the pure-Python DPT alternative.

---

## 2. Preprocessing: from a preprocessed AnnData to FateDriver input

Skip this section if you use the released datasets (§3.2) — they already ship
in the final format.

The starting point is one QC'd / normalised `.h5ad` per dataset, containing the
source **and** target cells together:

| Field                          | Meaning                                              |
|--------------------------------|------------------------------------------------------|
| `adata.X`                      | log-normalised (and scaled) expression               |
| `adata.layers['counts']`       | raw integer counts                                   |
| `adata.var['highly_variable']` | HVG flag                                             |
| `adata.obs[<group_key>]`       | cell-state label, e.g. `cell_type` ∈ {Fibroblast, iPSC} |
| `adata.obs[<cluster_key>]`     | `leiden`/`louvain` clusters (slingshot `--cluster_by cluster`) |
| `adata.obsm['X_umap']`         | UMAP embedding (slingshot path)                      |
| `adata.obsm['X_pca']`          | PCA embedding (DPT path; recomputed if absent)       |

### 2.1 Pseudotime — `scripts/compute_pseudotime.py`

Orders the cells along the transition and writes `cell_id,pseudotime`.

**Slingshot** (default). The AnnData is exported to a temporary directory, and `scripts/compute_pseudotime.R` fits Slingshot on the existing UMAP embedding. The cell-state labels (`--group_key`) are used as clusters, `--root_group` is the start cluster and `--end_group` (optional) is the end cluster: `slingshot(umap, clusterLabels = group, start.clus = root_group, end.clus = end_group)`. A cell's pseudotime is the mean of its per-lineage pseudotimes (`rowMeans(slingPseudotime(), na.rm = TRUE)`). Cells that belong to no lineage are dropped.

    conda activate FateDriver
    python scripts/compute_pseudotime.py \
        --h5ad       /path/GSE297234_GM00731_preprocessed.h5ad \
        --group_key  cell_type \
        --root_group Fibroblast \
        --end_group  iPSC \
        --method     slingshot \
        --rscript    "$(conda info --base)/envs/FateDriver-pseudotime/bin/Rscript" \
        --plot_prefix figures/GSE297234_GM00731 \
        --out_csv    /path/GSE297234_GM00731_pseudotime.csv

If you omit `--end_group`, Slingshot detects the end points itself and may return several lineages. `--cluster_by cluster` uses the `--cluster_key` clusters (default `leiden`) as `clusterLabels` instead of the cell-state labels. The start and end clusters are then the clusters that hold most of the root and end cells. `--plot_prefix` also writes the trajectory, pseudotime and pseudotime-per-cell-state figures as PNGs.

**DPT** (pure Python, no R). This uses scanpy diffusion pseudotime. The root cell is the cell of `--root_group` that lies furthest, in diffusion space, from the centre of the other cells:

    python scripts/compute_pseudotime.py \
        --h5ad       /path/GSE98567_preprocessed.h5ad \
        --group_key  group --root_group Control_uninfected \
        --method     dpt \
        --out_csv    /path/GSE98567_pseudotime.csv

The two methods give values on different scales. FateDriver only uses the pseudotime ordering to orient the cell trajectory graph. Useful flags: `--end_group`, `--cluster_by`, `--cluster_key`, `--umap_key`, `--keep_work_dir` (slingshot); `--num_dim`, `--n_neighbors`, `--n_dcs` (dpt); `--seed`.


### 2.2 Unified format — `scripts/prepare_dataset.py`

Combines the AnnData and the pseudotime table into the two files FateDriver
reads, keeping only the fields the model needs (so the published files stay
small):

```bash
python scripts/prepare_dataset.py \
    --src_h5ad       /path/GSE297234_GM00731_preprocessed.h5ad \
    --src_pseudotime /path/GSE297234_GM00731_pseudotime.csv \
    --obs_key        cell_type \
    --out_dir        data/GSE297234_GM00731
```

`--obs_key` names the obs column holding the source/target label; it is copied
to `obs['condition']`, which is what `--obs_column condition` refers to at
training time.

`scripts/prepare_all_datasets.sh` runs both steps for every published dataset
and documents the exact provenance of the released files:

`RAW_ROOT` (the directory holding the preprocessed `.h5ad` files) is required.

```bash
RAW_ROOT=/path/to/DATA COMPUTE_PSEUDOTIME=1 \
    RSCRIPT=/path/to/envs/FateDriver-pseudotime/bin/Rscript \
    bash scripts/prepare_all_datasets.sh
```

---

## 3. Input data

### 3.1 Format

Each dataset is a directory with two files:

```
data/<dataset>/
├── anndata.h5ad     # cells × genes AnnData
└── pseudotime.csv   # columns: cell_id, pseudotime
```

The `anndata.h5ad` must contain:

| Field                     | Meaning                                                        |
|---------------------------|----------------------------------------------------------------|
| `adata.X`                 | model input features (normalised / scaled expression)          |
| `adata.layers['counts']`  | raw integer counts (used by the ZINB reconstruction)           |
| `adata.var['highly_variable']` | boolean HVG flag (genes used by the model)                |
| `adata.obs['condition']`  | source/target label per cell (e.g. `D0`, `iPSC`)               |
| `adata.obs_names`         | cell barcodes, matching `cell_id` in `pseudotime.csv`          |

Gene sets (small CSVs with a `gene` column) live in `data/gene_sets/`:

* **driver genes** — known intervention/driver genes = positive labels
  (e.g. `iPSC_oskm_driver_human_genes.csv` → `POU5F1, SOX2, KLF4, MYC`).
* **downstream genes** — direct targets forming the soft-negative tier
  (e.g. `TRRUST_human_OSKM_targets.csv`).
* **NicheNet** signaling table (`nichenet_gr_signaling_net*.csv`) used to build
  the gene graph.

`data/datasets.tsv` maps each dataset to its source/target labels, obs column,
gene-set files, AnnData / pseudotime paths and pseudotime column
(`pseudotime_key`). All paths are relative to `data/`.

### 3.2 Download the released data

The published datasets and the full gene-set / NicheNet tables are archived on
Zenodo (they are too large for git):

> **DOI [10.5281/zenodo.21771206](https://doi.org/10.5281/zenodo.21771206)**
> — <https://zenodo.org/records/21771206>

```bash
# from the repository root: all datasets + gene sets
bash scripts/download_data.sh

# or a subset
bash scripts/download_data.sh GSE98567 gene_sets
```

The record holds one zip per dataset plus `gene_sets.zip`. Each zip contains a
single top-level directory, so downloading manually just means unzipping every
archive into `data/`:

```bash
unzip GSE242423.zip -d data/     # -> data/GSE242423/{anndata.h5ad,pseudotime.csv}
unzip gene_sets.zip -d data/     # -> data/gene_sets/*.csv
```

Zenodo record → repository layout:

```
GSE181451_Baseline_MEF.zip   →  data/GSE181451_Baseline_MEF/  anndata.h5ad, pseudotime.csv
GSE181451_Primed_MEF.zip     →  data/GSE181451_Primed_MEF/    anndata.h5ad, pseudotime.csv
GSE242423.zip                →  data/GSE242423/               anndata.h5ad, pseudotime.csv
GSE297234_GM00731.zip        →  data/GSE297234_GM00731/       anndata.h5ad, pseudotime.csv
GSE297234_GM23815.zip        →  data/GSE297234_GM23815/       anndata.h5ad, pseudotime.csv
GSE98567.zip                 →  data/GSE98567/                anndata.h5ad, pseudotime.csv
GSE237173.zip                →  data/GSE237173/               <SRC>_Cont-aVEC_Dox_(root_<SRC>_Lineage1).h5ad
                                                              <SRC>_Cont-aVEC_Dox_(root_<SRC>_Lineage1)_pseudotime.csv
                                                              (SRC = CBC, EC, EE, PC, TA, TC)
gene_sets.zip                →  data/gene_sets/               GMT_driver_genes_mouse.csv
                                                              iPSC_oskm_driver_genes.csv
                                                              iPSC_oskm_driver_human_genes.csv
                                                              TRRUST_GMT_targets_mouse.csv
                                                              TRRUST_human_OSKM_targets.csv
                                                              TRRUST_mouse_OSKM_targets.csv
                                                              nichenet_gr_signaling_net.csv
                                                              nichenet_gr_signaling_net_human.csv
```

Released datasets: `GSE242423`, `GSE297234_GM00731`, `GSE297234_GM23815`,
`GSE98567`, `GSE181451_Primed_MEF`, `GSE181451_Baseline_MEF` and `GSE237173` —
the two `GSE297234` archives are the two donors (GM00731, GM23815), each an
independent dataset.

`GSE237173` (mouse intestine, doxycycline-induced OSKM) is used to rank the
candidate origins of the OSKM-induced aVEC population. Its archive holds six
lineage pairs, one per candidate source cell type — crypt-base columnar (CBC),
enterocyte (EC), enteroendocrine (EE), Paneth (PC), transit-amplifying (TA) and
tuft (TC) cells — each paired with the aVEC target and ordered by a Slingshot
pseudotime rooted at that source. In each pair the source/target label is
`obs['cellType']` and the pseudotime is the `pseudotime_lineage` column of the
pair's `_pseudotime.csv`. The pairs are registered as `GSE237173_<SRC>_aVEC`.

All of these are registered in `data/datasets.tsv` and can be run directly with
`scripts/run_fatedriver.sh`.

### 3.3 Bundled example

A small, fully self-contained example (a reduced GSE98567 with a filtered
NicheNet table, ~12 MB) ships in the repository under `data/example/` so you can
run FateDriver immediately without any download.

---

## 4. Running FateDriver

### 4.1 Quick start on the bundled example

```bash
conda activate FateDriver
python src/train_fatedriver.py \
    --anndata_path          data/example/anndata.h5ad \
    --pseudotime_csv_path   data/example/pseudotime.csv \
    --driver_genes_path     data/example/GMT_driver_genes_mouse.csv \
    --affected_genes_path   data/example/TRRUST_GMT_targets_mouse.csv \
    --gene_nichenet_df_path data/example/nichenet.csv \
    --source_cell_type Control_uninfected \
    --target_cell_type Reprogramming \
    --obs_column condition \
    --dataset example \
    --output_dir outputs/example \
    --seed 42 --device_num 0
```

The example converges in seconds. Its top-ranked genes are exactly the three
known GMT drivers, e.g.:

```
gene,score
Mef2c,4.205942
Tbx5,3.939906
Gata4,3.673461
...
```

### 4.2 Reproduce a published dataset

`scripts/run_fatedriver.sh` reads `data/datasets.tsv` and launches training with
the exact paper hyper-parameters (all defaults):

```bash
# bash scripts/run_fatedriver.sh <dataset> [seed] [device_num] [extra train_fatedriver.py args...]
PYTHON_BIN=$(which python) bash scripts/run_fatedriver.sh GSE242423 42 0

# GSE237173: one run per candidate source -> aVEC pair
for src in CBC EC EE PC TA TC; do
    PYTHON_BIN=$(which python) bash scripts/run_fatedriver.sh GSE237173_${src}_aVEC 42 0
done
```

Any further arguments are passed on to `src/train_fatedriver.py`, e.g. a short
test run: `bash scripts/run_fatedriver.sh GSE98567 42 0 --n_epochs 10
--stage1_end_epoch 4 --stage2_end_epoch 6`.

The paper reports the mean over **seeds 42–46**; run the five seeds and average
the resulting `predictions.csv` rankings.

### 4.3 Key hyper-parameters (defaults = paper configuration)

| Flag | Default | Meaning |
|------|---------|---------|
| `--hidden_dim` | 1024 | embedding dimension |
| `--n_components` | 50 | TruncatedSVD cell features |
| `--num_cell_layers` / `--num_gene_layers` | 2 / 2 | GCN depth |
| `--recon_loss` | zinb | reconstruction likelihood (decoder `exp`) |
| `--n_epochs` | 200 | with stages at 50 / 70 |
| `--early_stopping_patience` | 10 | on driver+downstream AUPRC (stage 3) |
| `--lr` / `--weight_decay` | 1e-4 / 1e-4 | Adam |
| `--set_size` | 128 | cells per source/target episode |
| `--num_neg_samples` | 500 | negatives per ranking step |

Run `python src/train_fatedriver.py --help` for the full list.

---

## 5. Output

Each run creates `outputs/<dataset>/run_<timestamp>_dataset-<dataset>_seed-<seed>/`:

```
├── predictions.csv                 # ← main result: gene,score (ranked, best first)
├── predictions.npz                 # same, as arrays
├── test_metrics.txt                # hit@K, prec@K, mrr on held-out drivers
├── meta.json                       # full run configuration
├── training_log.csv                # per-epoch losses & validation metrics
├── model_weights/
│   ├── cell_encoder.pt  gene_encoder.pt  gene_feature_projector.pt
│   ├── set_encoder.pt   ranking_head.pt
│   ├── driver_prior_model.pt  driver_prior_metadata.json   # learned prior weights
│   └── recon_params.pt
├── preprocessing/svd_model.pkl     # fitted SVD + scaler (for transfer/inference)
├── embeddings/
│   ├── gene_embeddings/gene_embedding.npz     # gene embedding + UMAP
│   └── cell_embeddings/cell_embedding.npz     # cell embedding + UMAP
├── tables/
│   ├── gene_umap.csv               # gene, umap1, umap2, is_driver, score
│   └── cell_umap.csv               # obs_name, umap1, umap2, condition, group
├── unknown_driver_candidates.csv   # top-ranked non-annotated candidate drivers
├── cell_graph_stats.json  graph_stats.json  final_graph_stats.json
└── graph_pruning.csv
```

**`predictions.csv`** is the primary output — a full ranking of all model genes:

```
gene,score
POU5F1,5.12
SOX2,4.87
...
```

**`test_metrics.txt`** reports driver-recovery metrics on the held-out driver
genes:

```
hit@K: 1.000000
prec@K: 0.300000
mrr: 1.000000
```

---

## 6. Repository layout

```
.
├── environment.yml                 # conda environment (pinned)
├── environment_pseudotime.yml      # optional R env: Slingshot pseudotime
├── README.md
├── src/
│   ├── train_fatedriver.py         # entry point: training + inference
│   ├── model.py                    # network modules + ZINB/NB/MSE decoders
│   ├── utils.py                    # data, graphs, losses, sampling, metrics, IO
│   └── export_embeddings.py        # embedding + UMAP export
├── scripts/
│   ├── compute_pseudotime.py       # AnnData -> pseudotime.csv (slingshot | DPT)
│   ├── compute_pseudotime.R        # Slingshot trajectory + pseudotime
│   ├── prepare_dataset.py          # AnnData + pseudotime -> unified format
│   ├── prepare_all_datasets.sh     # rebuild every released dataset (provenance)
│   ├── run_fatedriver.sh           # train one dataset from datasets.tsv
│   ├── build_example.py            # build the bundled example
│   └── download_data.sh            # fetch the released data from Zenodo
├── data/
│   ├── datasets.tsv                # dataset registry
│   ├── gene_sets/                  # driver / downstream / NicheNet gene sets
│   └── example/                    # self-contained runnable example
└── checkpoints/                    # pretrained (seed 42) weights
```

---

## 7. Citation

If you use FateDriver, please cite the accompanying paper (details to be added
upon publication).
