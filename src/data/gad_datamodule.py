import os
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import requests
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Amazon,
    EllipticBitcoinDataset,
    HeterophilousGraphDataset,
    Planetoid,
    TUDataset,
)
from torch_geometric.loader import DataLoader


_PLANETOID_NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
_AMAZON_NAMES = {
    "amazon_computers": "Computers",
    "amazon-computers": "Computers",
    "amazoncomputers": "Computers",
    "computers": "Computers",
    "amazon_photo": "Photo",
    "amazon-photo": "Photo",
    "amazonphoto": "Photo",
    "photo": "Photo",
}
_HETEROPHILOUS_NAMES = {
    "tolokers": "Tolokers",
    "questions": "Questions",
}
_ELLIPTIC_NAMES = {"elliptic", "elliptic_bitcoin", "elliptic-bitcoin"}
_TFINANCE_NAMES = {"tfinance", "t-finance", "t_finance"}


def _identify_dataset_source(name: str) -> str:
    """Choose a loader; PyGOD names are case-sensitive."""
    pygod_names = {"weibo", "reddit", "disney", "books", "enron"}
    if name in pygod_names or name.startswith("inj_") or name.startswith("gen_"):
        return "pygod"

    dgl_fraud_names = {"yelpchi", "yelp_chi", "fraud_yelp", "fraudyelp"}
    if name.lower() in dgl_fraud_names:
        return "dgl_fraud"

    if name.lower() in _PLANETOID_NAMES:
        return "planetoid"

    if name.lower() in _AMAZON_NAMES:
        return "amazon"

    if name.lower() in _HETEROPHILOUS_NAMES:
        return "pyg_heterophilous"

    if name.lower() in _ELLIPTIC_NAMES:
        return "pyg_elliptic"

    if name.lower() in _TFINANCE_NAMES:
        return "pyg_cached_tfinance"

    return "tudataset"


def _resolve_planetoid_name(name: str) -> str:
    try:
        return _PLANETOID_NAMES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown Planetoid dataset name: {name}") from exc


def _resolve_amazon_name(name: str) -> str:
    normalized = name.lower()
    if normalized == "amazon":
        raise ValueError(
            "The PyG Amazon benchmark has two variants. Use "
            "'amazon_computers' or 'amazon_photo'."
        )
    try:
        return _AMAZON_NAMES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown Amazon dataset name: {name}") from exc


def _resolve_heterophilous_name(name: str) -> str:
    try:
        return _HETEROPHILOUS_NAMES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown heterophilous dataset name: {name}") from exc


def load_pygod_data(name, cache_dir=None):
    """Load PyGOD dataset from GitHub repository."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.pygod', 'data')
    file_path = os.path.join(cache_dir, name + '.pt')
    zip_path = os.path.join(cache_dir, name + '.pt.zip')

    if os.path.exists(file_path):
        data = torch.load(file_path, weights_only=False)
    else:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if not os.path.exists(zip_path) or not zipfile.is_zipfile(zip_path):
            url = "https://github.com/pygod-team/data/raw/main/" + name + ".pt.zip"
            last_error = None
            for _ in range(3):
                try:
                    r = requests.get(url, stream=True, timeout=60)
                    if r.status_code == 200:
                        break
                    last_error = RuntimeError("Failed downloading url %s" % url)
                except requests.RequestException as exc:
                    last_error = exc
            else:
                raise RuntimeError(f"Failed downloading url {url}") from last_error
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # Skip HTTP keep-alive chunks.
                        f.write(chunk)
        shutil.unpack_archive(zip_path, cache_dir)
        data = torch.load(file_path, weights_only=False)
    return data


def load_tfinance_data(cache_dir: Optional[str] = None) -> Data:
    """Load the DGL-free T-Finance tensor bundle produced by the converter."""
    cache_root = Path(
        cache_dir or os.path.join(os.path.expanduser("~"), ".pygod", "data")
    )
    candidates = (
        cache_root / "tfinance.pt",
        cache_root / "tfinance" / "processed" / "tfinance.pt",
    )
    file_path = next((path for path in candidates if path.exists()), None)
    if file_path is None:
        expected = candidates[0]
        raise FileNotFoundError(
            f"T-Finance cache not found at {expected}. Convert the official DGL graph with "
            "scripts/convert_tfinance_dgl.py before training."
        )

    payload = torch.load(file_path, map_location="cpu", weights_only=True)
    if isinstance(payload, Data):
        data = payload
    elif isinstance(payload, dict):
        missing = {"x", "edge_index", "y"} - payload.keys()
        if missing:
            raise ValueError(f"T-Finance cache is missing fields: {sorted(missing)}")
        if payload.get("dataset") not in (None, "tfinance"):
            raise ValueError(
                f"Unexpected dataset marker in T-Finance cache: {payload.get('dataset')}"
            )
        data = Data(
            x=payload["x"].float(),
            edge_index=payload["edge_index"].long(),
            y=payload["y"].long(),
        )
    else:
        raise TypeError(f"Unsupported T-Finance cache payload: {type(payload)}")

    if data.x is None or data.x.ndim != 2:
        raise ValueError("T-Finance node features must have shape [num_nodes, num_features].")
    if data.edge_index.ndim != 2 or data.edge_index.size(0) != 2:
        raise ValueError("T-Finance edge_index must have shape [2, num_edges].")
    if data.y is None or data.y.reshape(-1).numel() != int(data.num_nodes):
        raise ValueError("T-Finance labels must contain one value per node.")
    data.y = data.y.reshape(-1).long()
    label_values = sorted(data.y.unique().tolist())
    if label_values != [0, 1]:
        raise ValueError(f"T-Finance labels must be binary, got values={label_values}.")
    data.has_anomaly_labels = True
    return data


def _get_dgl_node_data(graph: Any, keys: Sequence[str]) -> torch.Tensor:
    for key in keys:
        if key in graph.ndata:
            return graph.ndata[key]
    raise KeyError(f"Could not find any of node data keys {list(keys)} in DGL graph.")


def _dgl_graph_to_pyg_data(graph: Any) -> Data:
    try:
        import dgl
    except ModuleNotFoundError as exc:
        raise RuntimeError("DGL is required to convert a DGL graph to PyG Data.") from exc

    if not graph.is_homogeneous:
        candidate_node_keys = [key for key in ("feature", "feat", "x", "label", "y") if key in graph.ndata]
        graph = dgl.to_homogeneous(graph, ndata=candidate_node_keys)

    x = _get_dgl_node_data(graph, ("feature", "feat", "x")).float()
    y = _get_dgl_node_data(graph, ("label", "y")).long().reshape(-1)
    src, dst = graph.edges(order="eid")
    edge_index = torch.stack([src.long(), dst.long()], dim=0)
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=int(graph.num_nodes()))


def _download_with_retries(url: str, output_path: Path, attempts: int = 3) -> None:
    last_error = None
    for _ in range(attempts):
        try:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                with output_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                return
            last_error = RuntimeError(f"Failed downloading url {url}")
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(f"Failed downloading url {url}") from last_error


def _find_yelpchi_mat(cache_root: Path) -> Path | None:
    for candidate in [
        cache_root / "YelpChi.mat",
        cache_root / "yelpchi_raw" / "YelpChi.mat",
        cache_root / "dgl" / "YelpChi.mat",
    ]:
        if candidate.exists():
            return candidate
    matches = list(cache_root.glob("**/YelpChi.mat"))
    return matches[0] if matches else None


def _load_yelpchi_from_mat(mat_path: Path) -> Data:
    from scipy import io
    from torch_geometric.utils import coalesce

    raw = io.loadmat(mat_path)
    x_raw = raw["features"]
    x = torch.as_tensor(np.asarray(x_raw.todense() if hasattr(x_raw, "todense") else x_raw), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(raw["label"]).reshape(-1), dtype=torch.long)
    edge_indices = []
    for relation in ("net_rsr", "net_rtr", "net_rur"):
        adj = raw[relation].tocoo()
        edge_indices.append(
            torch.stack(
                [
                    torch.as_tensor(adj.row, dtype=torch.long),
                    torch.as_tensor(adj.col, dtype=torch.long),
                ],
                dim=0,
            )
        )
    edge_index = torch.cat(edge_indices, dim=1)
    edge_index = coalesce(edge_index, num_nodes=int(x.size(0)))
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=int(x.size(0)))


def load_yelpchi_data(cache_dir: Optional[str] = None) -> Data:
    """Load the PyG cache or convert the original YelpChi MAT archive."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".pygod", "data")
    cache_root = Path(cache_dir)
    file_path = cache_root / "yelpchi.pt"
    if file_path.exists():
        data = torch.load(file_path, weights_only=False)
        if not isinstance(data, Data):
            raise TypeError(f"Expected cached YelpChi PyG Data at {file_path}, got {type(data)}.")
        return data

    cache_root.mkdir(parents=True, exist_ok=True)
    mat_path = _find_yelpchi_mat(cache_root)
    if mat_path is None:
        zip_path = cache_root / "FraudYelp.zip"
        if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
            try:
                _download_with_retries("https://data.dgl.ai/dataset/FraudYelp.zip", zip_path)
            except RuntimeError as exc:
                raise RuntimeError(
                    "YelpChi is not cached and automatic download failed. Place either "
                    f"a converted PyG Data file at {file_path}, the original YelpChi.mat "
                    f"under {cache_root}, or FraudYelp.zip at {zip_path}."
                ) from exc
        raw_dir = cache_root / "yelpchi_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(zip_path, raw_dir)
        mat_path = _find_yelpchi_mat(cache_root)

    if mat_path is None:
        raise RuntimeError(
            f"Could not find YelpChi.mat under {cache_root}. Place YelpChi.mat or "
            f"a converted PyG Data file at {file_path}."
        )

    data = _load_yelpchi_from_mat(mat_path)
    torch.save(data, file_path)
    return data


def collapse_anomaly_types(dataset, level='node', indicator='y'):
    if level == 'node':
        for data in dataset:
            data[indicator] = (data[indicator] >= 1).int()


def ignore_anomaly_types(dataset, level='node', indicator='y', type_ids=2):
    if type(type_ids) == int:
        type_ids = [type_ids]
    if level == 'node':
        for data in dataset:
            for idx in type_ids:
                data[indicator][data[indicator] == idx] = 0


def _normalize_graph_features(data_list: List[Data], method: str) -> None:
    method = (method or "none").lower()
    if method in {"none", "false", "off"}:
        return

    features = [graph.x.float() for graph in data_list if graph.x is not None]
    if not features:
        return

    all_x = torch.cat(features, dim=0)
    eps = 1e-6

    if method == "zscore":
        center = all_x.mean(dim=0, keepdim=True)
        scale = all_x.std(dim=0, keepdim=True).clamp_min(eps)
        transform = lambda x: (x.float() - center.to(x.device)) / scale.to(x.device)
    elif method == "robust":
        center = all_x.median(dim=0, keepdim=True).values
        scale = (all_x - center).abs().median(dim=0, keepdim=True).values.clamp_min(eps)
        transform = lambda x: (x.float() - center.to(x.device)) / scale.to(x.device)
    elif method == "log_zscore":
        min_x = all_x.min(dim=0, keepdim=True).values
        shifted = torch.log1p((all_x - min_x).clamp_min(0.0))
        center = shifted.mean(dim=0, keepdim=True)
        scale = shifted.std(dim=0, keepdim=True).clamp_min(eps)
        transform = lambda x: (
            torch.log1p((x.float() - min_x.to(x.device)).clamp_min(0.0)) - center.to(x.device)
        ) / scale.to(x.device)
    elif method == "row_l2":
        transform = lambda x: x.float() / x.float().norm(dim=1, keepdim=True).clamp_min(eps)
    elif method == "log_row_l2":
        min_x = all_x.min(dim=0, keepdim=True).values
        def transform(x):
            logged = torch.log1p((x.float() - min_x.to(x.device)).clamp_min(0.0))
            return logged / logged.norm(dim=1, keepdim=True).clamp_min(eps)
    else:
        raise ValueError(f"Unknown feature_normalization method: {method}")

    for graph in data_list:
        if graph.x is not None:
            graph.x = transform(graph.x).to(torch.float32)


def _apply_label_mode(data_list: List[Data], mode: str, source: str) -> None:
    mode = (mode or "auto").lower()
    if mode == "auto":
        mode = "class_to_attr" if source in {"planetoid", "amazon"} else "anomaly"

    if mode in {"anomaly", "default"}:
        return
    if mode not in {"class_to_attr", "no_anomaly", "none"}:
        raise ValueError(f"Unknown label_mode: {mode}")

    for graph in data_list:
        if getattr(graph, "y", None) is not None:
            graph.class_y = graph.y.detach().clone()
        graph.y = torch.zeros(int(graph.num_nodes), dtype=torch.long)
        graph.has_anomaly_labels = False


def _prepare_source_labels(data_list: List[Data], source: str) -> None:
    """Normalize source-specific labels before generic anomaly handling.

    Elliptic uses ``0=licit``, ``1=illicit``, and ``2=unknown``. Unknown
    transactions remain available to the unsupervised model and graph
    propagation, but must never enter label-based evaluation metrics. When
    PyG's official temporal test mask is present, evaluation is restricted to
    known transactions from time steps 35--49.
    """
    if source != "pyg_elliptic":
        return

    for graph in data_list:
        y = getattr(graph, "y", None)
        if y is None:
            raise ValueError("Elliptic data is missing node labels.")
        y = y.reshape(-1).long()
        if y.numel() != int(graph.num_nodes):
            raise ValueError(
                f"Elliptic label count {y.numel()} does not match num_nodes={graph.num_nodes}."
            )
        invalid = ~torch.isin(y, y.new_tensor([0, 1, 2]))
        if invalid.any():
            values = sorted(y[invalid].unique().tolist())
            raise ValueError(f"Unexpected Elliptic label values: {values}")

        graph.original_y = y.clone()
        known_label_mask = y != 2
        graph.known_label_mask = known_label_mask
        test_mask = getattr(graph, "test_mask", None)
        if test_mask is None:
            graph.eval_mask = known_label_mask.clone()
        else:
            test_mask = test_mask.reshape(-1).bool()
            if test_mask.numel() != int(graph.num_nodes):
                raise ValueError(
                    "Elliptic test mask count "
                    f"{test_mask.numel()} does not match num_nodes={graph.num_nodes}."
                )
            graph.eval_mask = known_label_mask & test_mask
            if not graph.eval_mask.any():
                raise ValueError("Elliptic temporal test mask has no known labels.")
        graph.y = (y == 1).long()
        graph.has_anomaly_labels = True


def _load_and_normalize_dataset(
    name: str,
    cache_dir: Optional[str] = None,
    feature_normalization: str = "none",
    preserve_raw_features: bool = False,
    label_mode: str = "auto",
) -> List[Data]:
    """Load PyG graphs, normalize features, and apply the requested label mode."""
    source = _identify_dataset_source(name)

    if source == "pygod":
        raw_data = load_pygod_data(name, cache_dir)
    elif source == "pyg_cached_tfinance":
        raw_data = load_tfinance_data(cache_dir)
    elif source == "dgl_fraud":
        raw_data = load_yelpchi_data(cache_dir)
    elif source == "planetoid":
        root = cache_dir or os.path.join(os.path.expanduser('~'), '.pyg_data', 'Planetoid')
        dataset = Planetoid(root=root, name=_resolve_planetoid_name(name))
        raw_data = dataset
    elif source == "amazon":
        root = cache_dir or os.path.join(os.path.expanduser('~'), '.pyg_data', 'Amazon')
        dataset = Amazon(root=root, name=_resolve_amazon_name(name))
        raw_data = dataset
    elif source == "pyg_heterophilous":
        root = Path(cache_dir or os.path.join(os.path.expanduser("~"), ".pyg_data")) / "heterophilous"
        dataset = HeterophilousGraphDataset(
            root=str(root),
            name=_resolve_heterophilous_name(name),
        )
        raw_data = dataset
    elif source == "pyg_elliptic":
        root = Path(cache_dir or os.path.join(os.path.expanduser("~"), ".pyg_data")) / "elliptic"
        raw_data = EllipticBitcoinDataset(root=str(root))
    elif source == "tudataset":
        root = cache_dir or os.path.join(os.path.expanduser('~'), '.pyg_data', 'TUDataset')
        dataset = TUDataset(root=root, name=name)
        raw_data = dataset
    else:
        raise ValueError(f"Unknown dataset source: {source}")

    if isinstance(raw_data, Data):
        data_list = [raw_data]
    elif isinstance(raw_data, Sequence) and not isinstance(raw_data, str):
        data_list = list(raw_data)
    elif hasattr(raw_data, '__len__') and hasattr(raw_data, '__getitem__'):
        data_list = [raw_data[i] for i in range(len(raw_data))]
    else:
        raise TypeError(f"Unsupported data type from {source}: {type(raw_data)}")

    if not data_list:
        raise ValueError(f"Dataset '{name}' is empty")

    _prepare_source_labels(data_list, source)

    if preserve_raw_features:
        for graph in data_list:
            if graph.x is not None:
                graph.x_raw = graph.x.float().clone()

    _normalize_graph_features(data_list, feature_normalization)
    _apply_label_mode(data_list, label_mode, source)

    return data_list


class _GraphDataset(Dataset):
    """Dataset wrapper for a list of PyG Data objects."""

    def __init__(self, data_list: Sequence[Any]):
        if not data_list:
            raise ValueError("Graph dataset received an empty data list.")
        self._data_list = []
        for item in data_list:
            if not isinstance(item, Data):
                raise TypeError(f"Expected PyG Data instances, got {type(item)}")
            self._data_list.append(item)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]


class GADDataModule(LightningDataModule):
    """Load graph anomaly datasets for unsupervised training.

    Supports PyGOD, PyG benchmarks, converted T-Finance and YelpChi
    caches, and TUDataset. Feature normalization is shared across
    graphs; Elliptic unknown labels are excluded from evaluation.
    """

    def __init__(
        self,
        dataset: str = "disney",
        cache_dir: Optional[str] = None,
        feature_normalization: str = "none",
        preserve_raw_features: bool = False,
        label_mode: str = "auto",
        batch_size: int = 1,
        num_workers: int = 0,
        unify_anomaly_types: bool = False,
        attr_anomaly=True,
        struct_anomaly=True,
        combined_anomaly=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data = None
        self.dataset = None
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None
        self.unify_anomaly_types = unify_anomaly_types
        self.ignore_anomaly_types = []
        if not attr_anomaly:
            self.ignore_anomaly_types.append(1)
        if not struct_anomaly:
            self.ignore_anomaly_types.append(2)
        if not combined_anomaly:
            self.ignore_anomaly_types.append(3)

    @property
    def num_features(self) -> int:
        return int(self.data.num_node_features) if self.data is not None else 0

    def prepare_data(self):
        # Loaders download missing datasets during setup.
        pass

    def setup(self, stage: Optional[str] = None):
        if self.dataset is None:
            data_list = _load_and_normalize_dataset(
                self.hparams.dataset,
                cache_dir=self.hparams.cache_dir,
                feature_normalization=self.hparams.feature_normalization,
                preserve_raw_features=self.hparams.preserve_raw_features,
                label_mode=self.hparams.label_mode,
            )

            self.dataset = _GraphDataset(data_list)
            self.data = self.dataset[0]
        if self.unify_anomaly_types:
            collapse_anomaly_types(self.dataset)
        if len(self.ignore_anomaly_types) > 0:
            ignore_anomaly_types(self.dataset, type_ids=self.ignore_anomaly_types)

        self._train_ds = self.dataset
        self._val_ds = self.dataset
        self._test_ds = self.dataset

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
