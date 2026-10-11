# RNR-GAD

Implementation and fixed configurations for the nine evaluated datasets.

## Environment and data

The verified environment uses Python 3.10, PyTorch 2.8.0 (CUDA 12.9), Lightning
2.6.0, and PyG 2.7.0. Install PyTorch and the PyG extensions for your CUDA platform,
then install the packages listed in `requirements.txt`. The verified extension
versions are `torch-scatter 2.1.2+pt28cu129` and `torch-sparse 0.6.18+pt28cu129`.

Prepare the fixed PyG data files in `data/pygod/`: `books.pt`, `disney.pt`,
`enron.pt`, `reddit.pt`, `inj_cora.pt`, `inj_flickr.pt`, `weibo.pt`, `yelpchi.pt`,
and `tfinance.pt`. The existing Books data are retained. For another cache
location, pass `data.cache_dir=/absolute/path/to/cache` to the training command.

## Main experiments

Each multi-seed command runs seeds 0 through 9 and evaluates the final training
state, following the adopted main-experiment protocol.

```sh
python -m src.multi_train experiment=rnrgad_disney
python -m src.multi_train experiment=rnrgad_tfinance
```

The dataset launcher runs the selected experiments and records configuration,
source fingerprints, per-seed predictions, metrics, and checkpoints:

```sh
python scripts/run_rnrgad_multiseed.py --datasets books disney --gpus 0
```

An omitted `--datasets` runs all nine configurations. Choose an idle GPU and a
fresh output directory for a new batch. The `--dry-run` option writes a plan and
configuration snapshots without training.

| Dataset | Stage 1 + Stage 2 epochs | Maximum radius | Feature normalization |
| --- | ---: | ---: | --- |
| books | 40 + 10 | 2.00 | log_zscore |
| disney | 40 + 10 | 0.70 | zscore |
| enron | 40 + 10 | 1.00 | zscore |
| reddit | 100 + 10 | 1.00 | none |
| inj_cora | 40 + 10 | 0.05 | none |
| inj_flickr | 40 + 10 | 0.05 | none |
| weibo | 40 + 10 | 0.05 | none |
| yelpchi | 40 + 10 | 0.05 | zscore |
| tfinance | 5 + 5 | 1.00 | log_row_l2 |

## Capacity and role controls

Use the selected core dataset's complete configuration and add one intervention:

```sh
python -m src.multi_train experiment=rnrgad_disney ablation=scalar_gate
python -m src.multi_train experiment=rnrgad_disney ablation=wo_direct_anchor
python -m src.multi_train experiment=rnrgad_disney ablation=wo_graph_evidence
python -m src.multi_train experiment=rnrgad_disney ablation=wo_reliability
```

These controls use the same phase schedule and reseeding rule as the selected
dataset. Model files with versioned internal names provide inherited runtime
helpers used by the current method.

## Projection radius and checkpoint evaluation

The fixed maximum radius follows the empirical graph-count schedule documented
in the paper. A node's radius is `epsilon_max * (1 - confidence)`.

```sh
python scripts/resolve_projection_radius.py --nodes 39357 --stored-directed-edges 42445086 --features 10 --explain
python -m src.eval experiment=rnrgad_disney ckpt_path=/absolute/path/to/last.ckpt
```

Compose the same dataset and model configuration used to create the checkpoint.
