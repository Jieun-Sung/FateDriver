#!/usr/bin/env Rscript
# ============================================================================
# compute_pseudotime.R
#
# Slingshot trajectory + pseudotime for one FateDriver dataset
# (somatic cell -> iPSC).
#
# Same convention as the analysis notebooks/scripts:
#   - trajectory fitted on the precomputed UMAP embedding
#   - clusterLabels = cell state (`group`), start.clus = root group
#     (e.g. Fibroblast), optional end.clus (e.g. iPSC)
#   - unified pseudotime = rowMeans(slingPseudotime(), na.rm = TRUE)
#   - per-lineage pseudotime kept as pseudotime_lineage1, 2, ...
#
# It is normally called by `scripts/compute_pseudotime.py --method slingshot`,
# which exports the AnnData into <work_dir>:
#
#     barcodes.tsv   cell ids, one per line
#     cell_meta.csv  cell_id, group, cluster
#     umap.csv       cell_id, UMAP_1, UMAP_2
#     (counts.mtx / genes.tsv are not needed by Slingshot and are ignored)
#
# Standalone use:
#     Rscript scripts/compute_pseudotime.R \
#         --work_dir /tmp/export --out_csv data/MyDataset/pseudotime.csv \
#         --root_group Fibroblast --end_group iPSC
#
# Options:
#     --cluster_by  group (default) | cluster
#                   column of cell_meta.csv used as Slingshot clusterLabels.
#                   With `cluster`, the start/end clusters are the clusters
#                   holding most of the root/end group cells.
#     --end_group   optional terminal state; omit to let Slingshot detect
#                   end points itself (may give several lineages).
#
# Environment:
#     conda env create -f environment_pseudotime.yml
#     conda activate FateDriver-pseudotime
# ============================================================================
suppressPackageStartupMessages({
  library(slingshot)
  library(SingleCellExperiment)
})

# ---- minimal `--key value` argument parser --------------------------------
parse_args <- function(defaults) {
  argv <- commandArgs(trailingOnly = TRUE)
  if (length(argv) %% 2 != 0) stop("arguments must be given as --key value pairs")
  opts <- defaults
  if (length(argv) > 0) {
    keys <- sub("^--", "", argv[seq(1, length(argv), by = 2)])
    vals <- argv[seq(2, length(argv), by = 2)]
    for (i in seq_along(keys)) {
      if (!keys[i] %in% names(defaults)) stop(sprintf("unknown argument --%s", keys[i]))
      opts[[keys[i]]] <- vals[i]
    }
  }
  opts$seed <- as.integer(opts$seed)
  if (is.null(opts$work_dir) || is.null(opts$out_csv) || is.null(opts$root_group))
    stop("--work_dir, --out_csv and --root_group are required")
  if (!opts$cluster_by %in% c("group", "cluster"))
    stop("--cluster_by must be 'group' or 'cluster'")
  opts
}

opts <- parse_args(list(work_dir = NULL, out_csv = NULL, root_group = NULL,
                        end_group = NULL, cluster_by = "group", seed = 42,
                        plot_prefix = NULL))
set.seed(opts$seed)

# ---- load the exported tables ---------------------------------------------
wd <- opts$work_dir
cat(sprintf("[R] reading exchange files from %s\n", wd))

cells <- readLines(file.path(wd, "barcodes.tsv"))

cell_metadata <- read.csv(file.path(wd, "cell_meta.csv"), row.names = 1,
                          check.names = FALSE, stringsAsFactors = FALSE)
umap <- as.matrix(read.csv(file.path(wd, "umap.csv"), row.names = 1, check.names = FALSE))

missing <- setdiff(cells, intersect(rownames(cell_metadata), rownames(umap)))
if (length(missing) > 0)
  stop(sprintf("%d cells of barcodes.tsv are missing from cell_meta.csv / umap.csv",
               length(missing)))
cell_metadata <- cell_metadata[cells, , drop = FALSE]
umap          <- umap[cells, , drop = FALSE]

cat(sprintf("[R] %d cells\n", length(cells)))
cat("[R] groups:\n"); print(table(cell_metadata$group))

# ---- cluster labels + start / end clusters --------------------------------
labels <- setNames(as.character(cell_metadata[[opts$cluster_by]]), cells)

# Map a cell-state name to a clusterLabel: the label itself when clustering by
# group, otherwise the cluster that holds most cells of that state.
state_to_cluster <- function(state) {
  in_state <- cell_metadata$group == state
  if (!any(in_state)) stop(sprintf("no cell with group == '%s'", state))
  if (opts$cluster_by == "group") return(state)
  names(which.max(table(labels[in_state])))
}

start_clus <- state_to_cluster(opts$root_group)
end_clus   <- if (is.null(opts$end_group)) NULL else state_to_cluster(opts$end_group)
cat(sprintf("[R] clusterLabels = %s | start.clus = %s | end.clus = %s\n",
            opts$cluster_by, start_clus,
            if (is.null(end_clus)) "(auto)" else end_clus))

# ---- Slingshot on the precomputed UMAP -------------------------------------
# Reuse the embedding computed during preprocessing instead of a fresh one,
# so the trajectory matches the figures of the paper.
cat("[R] fitting slingshot...\n")
sds <- slingshot(umap, clusterLabels = labels,
                 start.clus = start_clus, end.clus = end_clus)

pt_mat <- slingPseudotime(sds)            # cells x lineages (NA = not on lineage)
n_lin  <- ncol(pt_mat)
cat(sprintf("[R] %d lineage(s):\n", n_lin))
for (lin in names(slingLineages(sds)))
  cat(sprintf("     %s: %s\n", lin, paste(slingLineages(sds)[[lin]], collapse = " -> ")))

# unified pseudotime: mean over the lineages a cell belongs to
unified_pt <- rowMeans(pt_mat, na.rm = TRUE)

# ---- Write cell_id,pseudotime,pseudotime_lineage* --------------------------
out <- data.frame(cell_id = cells, pseudotime = as.numeric(unified_pt[cells]))
for (i in seq_len(n_lin))
  out[[paste0("pseudotime_lineage", i)]] <- as.numeric(pt_mat[cells, i])

n_na <- sum(!is.finite(out$pseudotime))
if (n_na > 0)
  cat(sprintf("[R] %d cells are not assigned to any lineage (NA pseudotime)\n", n_na))

dir.create(dirname(opts$out_csv), showWarnings = FALSE, recursive = TRUE)
write.csv(out, opts$out_csv, row.names = FALSE, quote = FALSE)
cat(sprintf("[R] wrote %s (%d cells)\n", opts$out_csv, nrow(out)))

# ---- Optional QC figures ---------------------------------------------------
if (!is.null(opts$plot_prefix) && requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  dir.create(dirname(opts$plot_prefix), showWarnings = FALSE, recursive = TRUE)

  df <- data.frame(UMAP_1 = umap[, 1], UMAP_2 = umap[, 2],
                   group = cell_metadata$group, pseudotime = out$pseudotime)
  curve_df <- do.call(rbind, lapply(names(slingCurves(sds)), function(lin) {
    crv <- slingCurves(sds)[[lin]]
    data.frame(UMAP_1 = crv$s[crv$ord, 1], UMAP_2 = crv$s[crv$ord, 2], lineage = lin)
  }))
  curve_layer <- geom_path(data = curve_df, aes(UMAP_1, UMAP_2, group = lineage),
                           inherit.aes = FALSE, colour = "black", linewidth = 1.2)

  p_ct <- ggplot(df, aes(UMAP_1, UMAP_2, colour = group)) +
    geom_point(size = 0.3, alpha = 0.7) + curve_layer +
    guides(colour = guide_legend(override.aes = list(size = 3))) +
    theme_classic(base_size = 12) + ggtitle("Trajectory coloured by cell state")
  ggsave(paste0(opts$plot_prefix, "_trajectory.png"), p_ct, width = 6, height = 5, dpi = 200)

  p_pt <- ggplot(df[order(df$pseudotime, na.last = FALSE), ],
                 aes(UMAP_1, UMAP_2, colour = pseudotime)) +
    geom_point(size = 0.3, alpha = 0.8) + curve_layer +
    scale_color_viridis_c(option = "C", na.value = "grey85") +
    theme_classic(base_size = 12) + ggtitle("Pseudotime")
  ggsave(paste0(opts$plot_prefix, "_pseudotime.png"), p_pt, width = 6, height = 5, dpi = 200)

  pt_df <- df[is.finite(df$pseudotime), ]
  p_vio <- ggplot(pt_df, aes(x = group, y = pseudotime, fill = group)) +
    geom_violin(trim = FALSE, alpha = 0.7) +
    geom_boxplot(width = 0.1, fill = "white", outlier.size = 0.3) +
    labs(title = "Pseudotime distribution by cell state", x = NULL) +
    theme_bw(base_size = 12) + theme(legend.position = "none")
  ggsave(paste0(opts$plot_prefix, "_distribution.png"), p_vio, width = 6, height = 4, dpi = 200)

  cat(sprintf("[R] figures written with prefix %s\n", opts$plot_prefix))
}
