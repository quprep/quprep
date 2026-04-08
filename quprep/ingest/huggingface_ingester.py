"""HuggingFace Datasets ingestion — tabular, image, text, and graph data."""

from __future__ import annotations

import numpy as np

from quprep.core.dataset import Dataset
from quprep.ingest.csv_ingester import _detect_feature_types

_SUPPORTED_MODALITIES = ("tabular", "image", "text", "graph")

# HuggingFace feature class names that are detected but not yet supported.
# Maps feature class name → human-readable modality name for error messages.
_UNSUPPORTED_FEATURES: dict[str, str] = {
    "Audio": "audio",
    "Video": "video",
}


class HuggingFaceIngester:
    """
    Load a HuggingFace dataset into a Dataset.

    Automatically detects the dataset modality (tabular, image, text) from the
    HuggingFace feature schema.  Pass ``modality`` explicitly to override
    detection or to load graph datasets.

    Requires ``pip install quprep[huggingface]``.

    Parameters
    ----------
    split : str
        Dataset split to load (e.g. ``"train"``, ``"test"``). Default ``"train"``.
    modality : str
        One of ``"auto"`` (default), ``"tabular"``, ``"image"``, ``"text"``,
        ``"graph"``.  When ``"auto"`` the modality is inferred from the
        dataset's feature schema.
    target_columns : str or list of str, optional
        Column(s) to treat as labels.  Works for all modalities.
    numeric_only : bool
        *Tabular only.* Drop non-numeric columns (default ``True``).
    image_column : str, optional
        Image column name.  Auto-detected when ``modality="auto"``/``"image"``.
    image_size : tuple of (int, int)
        ``(height, width)`` to resize images before flattening. Default ``(28, 28)``.
    grayscale : bool
        Convert images to grayscale (default ``True``).
    normalize : bool
        Divide pixel values by 255 (default ``True``).
    text_column : str, optional
        Text column name.  Auto-detected when ``modality="auto"``/``"text"``.
    text_method : str
        ``"tfidf"`` (default, no extra deps) or ``"sentence_transformers"``.
    max_features : int or None
        Max TF-IDF vocabulary size (default 512).
    text_model : str
        sentence-transformers model name. Default ``"all-MiniLM-L6-v2"``.
    edge_index_column : str
        *Graph only.* Column containing edge indices in COO format ``[2, E]``.
        Default ``"edge_index"``.
    node_feature_column : str
        *Graph only.* Column containing node feature matrix. Default ``"x"``.
    n_graph_features : int or None
        Pad/truncate graph feature vectors to this length. Default: auto.
    token : str or bool, optional
        HuggingFace auth token for gated datasets.

    Examples
    --------
    Tabular (auto-detected)::

        ds = HuggingFaceIngester(split="train", target_columns="label").load(
            "imodels/credit-card"
        )

    Image (auto-detected from HF feature schema)::

        ds = HuggingFaceIngester(
            split="train", target_columns="label", image_size=(28, 28)
        ).load("ylecun/mnist")

    Text (auto-detected)::

        ds = HuggingFaceIngester(
            split="train", target_columns="label", text_method="tfidf"
        ).load("imdb")

    Graph (explicit)::

        ds = HuggingFaceIngester(
            modality="graph", split="train", target_columns="y"
        ).load("graphs-datasets/ogbg-molhiv")
    """

    def __init__(
        self,
        split: str = "train",
        modality: str = "auto",
        target_columns: str | list[str] | None = None,
        # tabular
        numeric_only: bool = True,
        # image
        image_column: str | None = None,
        image_size: tuple[int, int] = (28, 28),
        grayscale: bool = True,
        normalize: bool = True,
        # text
        text_column: str | None = None,
        text_method: str = "tfidf",
        max_features: int | None = 512,
        text_model: str = "all-MiniLM-L6-v2",
        # graph
        edge_index_column: str = "edge_index",
        node_feature_column: str = "x",
        n_graph_features: int | None = None,
        # auth
        token: str | bool | None = None,
    ):
        valid = ("auto",) + _SUPPORTED_MODALITIES
        if modality not in valid:
            raise ValueError(f"modality must be one of {valid}, got '{modality}'")
        self.split = split
        self.modality = modality
        self.target_columns = target_columns
        self.numeric_only = numeric_only
        self.image_column = image_column
        self.image_size = image_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.text_column = text_column
        self.text_method = text_method
        self.max_features = max_features
        self.text_model = text_model
        self.edge_index_column = edge_index_column
        self.node_feature_column = node_feature_column
        self.n_graph_features = n_graph_features
        self.token = token

    def load(self, dataset_name: str, config_name: str | None = None) -> Dataset:
        """
        Load a HuggingFace dataset by name.

        Parameters
        ----------
        dataset_name : str
            HuggingFace dataset identifier, e.g. ``"imdb"``, ``"ylecun/mnist"``.
        config_name : str, optional
            Dataset configuration/subset name.  E.g. ``"en"`` for multilingual
            datasets (maps to the ``name`` argument in ``load_dataset``).

        Returns
        -------
        Dataset

        Raises
        ------
        ImportError
            If ``datasets`` (or Pillow / sentence-transformers / networkx) is
            not installed.
        NotImplementedError
            If the dataset contains a modality QuPrep does not yet support
            (e.g. audio, video).
        ValueError
            If no usable columns remain after filtering.
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "HuggingFaceIngester requires the 'datasets' package. "
                "Install it with: pip install quprep[huggingface]"
            ) from e

        kwargs: dict = {"split": self.split}
        if config_name is not None:
            kwargs["name"] = config_name
        if self.token is not None:
            kwargs["token"] = self.token

        hf_dataset = load_dataset(dataset_name, **kwargs)

        # resolve modality
        modality = self.modality
        detected_col: str | None = None
        if modality == "auto":
            modality, detected_col = self._detect_modality(
                hf_dataset.features, dataset_name
            )

        meta_base = {
            "source": f"huggingface:{dataset_name}",
            "split": self.split,
            "config": config_name,
            "modality": modality,
        }

        if modality == "tabular":
            import pandas as pd
            return self._load_tabular(hf_dataset, dataset_name, config_name, pd)
        if modality == "image":
            return self._load_image(hf_dataset, dataset_name, meta_base, detected_col)
        if modality == "text":
            return self._load_text(hf_dataset, dataset_name, meta_base, detected_col)
        if modality == "graph":
            return self._load_graph(hf_dataset, dataset_name, meta_base)
        # Should never reach here — validated in __init__
        raise ValueError(f"Unknown modality '{modality}'")  # pragma: no cover

    # ------------------------------------------------------------------
    # Modality detection
    # ------------------------------------------------------------------

    def _detect_modality(
        self, features, dataset_name: str
    ) -> tuple[str, str | None]:
        """Return ``(modality, primary_column)`` from the HF feature schema."""
        image_cols: list[str] = []
        unsupported_cols: dict[str, str] = {}  # col → modality name
        text_cols: list[str] = []
        numeric_cols: list[str] = []

        label_cols = set()
        if self.target_columns is not None:
            label_cols = (
                {self.target_columns}
                if isinstance(self.target_columns, str)
                else set(self.target_columns)
            )

        for col, feat in features.items():
            if col in label_cols:
                continue
            feat_cls = type(feat).__name__
            if feat_cls in _UNSUPPORTED_FEATURES:
                unsupported_cols[col] = _UNSUPPORTED_FEATURES[feat_cls]
            elif feat_cls == "Image":
                image_cols.append(col)
            elif feat_cls == "Value":
                if getattr(feat, "dtype", None) == "string":
                    text_cols.append(col)
                else:
                    numeric_cols.append(col)
            else:
                # Sequence, ClassLabel, Array2D, etc. — treated as numeric/tabular
                numeric_cols.append(col)

        # Unsupported modalities: raise only when there are NO usable alternatives
        if unsupported_cols and not image_cols and not numeric_cols:
            modality_names = sorted(set(unsupported_cols.values()))
            cols_str = list(unsupported_cols)
            raise NotImplementedError(
                f"Dataset '{dataset_name}' contains "
                f"{'/'.join(modality_names)} data (column(s): {cols_str}). "
                f"QuPrep currently supports: {list(_SUPPORTED_MODALITIES)}. "
                "Pass modality='tabular' to ignore unsupported columns and "
                "process any remaining numeric features."
            )

        if image_cols:
            return "image", image_cols[0]

        # Text: only auto-select when there are no numeric columns
        # (otherwise tabular drops string columns gracefully)
        if text_cols and not numeric_cols:
            return "text", text_cols[0]

        return "tabular", None

    # ------------------------------------------------------------------
    # Tabular handler
    # ------------------------------------------------------------------

    def _load_tabular(self, hf_dataset, dataset_name, config_name, pd) -> Dataset:
        df: pd.DataFrame = hf_dataset.to_pandas()

        labels = None
        if self.target_columns is not None:
            cols = (
                [self.target_columns]
                if isinstance(self.target_columns, str)
                else list(self.target_columns)
            )
            labels = df[cols].to_numpy()
            if labels.shape[1] == 1:
                labels = labels.ravel()
            df = df.drop(columns=cols)

        all_feature_names = list(df.columns)
        all_feature_types = _detect_feature_types(df)

        numeric_mask = [
            not (
                isinstance(df[col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(df[col])
                or df[col].dtype.name == "string"
            )
            for col in df.columns
        ]
        numeric_cols = [col for col, keep in zip(df.columns, numeric_mask) if keep]
        cat_cols = [col for col, keep in zip(df.columns, numeric_mask) if not keep]

        if not numeric_cols:
            raise ValueError(
                f"No numeric columns found in dataset '{dataset_name}' "
                f"(split='{self.split}'). Available columns: {all_feature_names}. "
                "Check target_columns or set modality='text'/'image' explicitly."
            )

        data = df[numeric_cols].to_numpy(dtype=float)
        numeric_types = [
            t for t, keep in zip(all_feature_types, numeric_mask) if keep
        ]
        categorical_data = (
            {} if self.numeric_only
            else {col: df[col].tolist() for col in cat_cols}
        )

        return Dataset(
            data=data,
            feature_names=numeric_cols,
            feature_types=numeric_types,
            categorical_data=categorical_data,
            metadata={
                "source": f"huggingface:{dataset_name}",
                "split": self.split,
                "config": config_name,
                "modality": "tabular",
                "original_columns": all_feature_names,
                "original_types": all_feature_types,
                "n_dropped_categorical": len(cat_cols),
            },
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Image handler
    # ------------------------------------------------------------------

    def _load_image(
        self,
        hf_dataset,
        dataset_name: str,
        meta_base: dict,
        detected_col: str | None,
    ) -> Dataset:
        try:
            from PIL import Image as PILImage
        except ImportError as e:
            raise ImportError(
                "Image datasets require Pillow. "
                "Install it with: pip install quprep[image]"
            ) from e

        col = self.image_column or detected_col
        if col is None:
            for c, feat in hf_dataset.features.items():
                if type(feat).__name__ == "Image":
                    col = c
                    break
        if col is None:
            raise ValueError(
                f"No Image column found in dataset '{dataset_name}'. "
                f"Available columns: {list(hf_dataset.features)}. "
                "Set image_column= explicitly."
            )

        labels = self._extract_labels(hf_dataset)
        raw_images = hf_dataset[col]

        arrays = []
        for raw in raw_images:
            img = self._coerce_pil(raw, PILImage)
            img = img.convert("L" if self.grayscale else "RGB")
            if self.image_size is not None:
                img = img.resize((self.image_size[1], self.image_size[0]))
            arr = np.asarray(img, dtype=float)
            if self.normalize:
                arr = arr / 255.0
            arrays.append(arr.flatten())

        data = np.stack(arrays)
        n_pixels = data.shape[1]

        return Dataset(
            data=data,
            feature_names=[f"px_{i}" for i in range(n_pixels)],
            feature_types=["continuous"] * n_pixels,
            metadata={
                **meta_base,
                "image_column": col,
                "image_size": self.image_size,
                "channels": 1 if self.grayscale else 3,
                "n_images": len(arrays),
            },
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Text handler
    # ------------------------------------------------------------------

    def _load_text(
        self,
        hf_dataset,
        dataset_name: str,
        meta_base: dict,
        detected_col: str | None,
    ) -> Dataset:
        col = self.text_column or detected_col
        if col is None:
            for c, feat in hf_dataset.features.items():
                if type(feat).__name__ == "Value" and getattr(feat, "dtype", None) == "string":
                    col = c
                    break
        if col is None:
            raise ValueError(
                f"No string column found in dataset '{dataset_name}'. "
                f"Available columns: {list(hf_dataset.features)}. "
                "Set text_column= explicitly."
            )

        texts: list[str] = hf_dataset[col]
        labels = self._extract_labels(hf_dataset)
        vectors = self._embed_text(texts)
        n_features = vectors.shape[1]

        return Dataset(
            data=vectors,
            feature_names=[f"emb_{i}" for i in range(n_features)],
            feature_types=["continuous"] * n_features,
            metadata={
                **meta_base,
                "text_column": col,
                "text_method": self.text_method,
                "n_samples": len(texts),
                "n_features": n_features,
            },
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Graph handler
    # ------------------------------------------------------------------

    def _load_graph(
        self,
        hf_dataset,
        dataset_name: str,
        meta_base: dict,
    ) -> Dataset:
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "Graph datasets require networkx. "
                "Install it with: pip install networkx"
            ) from e

        if self.edge_index_column not in hf_dataset.features:
            raise ValueError(
                f"Graph column '{self.edge_index_column}' not found in "
                f"'{dataset_name}'. Available columns: {list(hf_dataset.features)}. "
                "Set edge_index_column= to the column containing graph edges."
            )

        labels = self._extract_labels(hf_dataset)
        edge_indices = hf_dataset[self.edge_index_column]
        n_samples = len(hf_dataset)

        graphs = []
        for i in range(n_samples):
            ei = np.asarray(edge_indices[i])
            if ei.ndim == 2 and ei.shape[0] == 2:
                src, dst = ei[0], ei[1]
            elif ei.ndim == 2 and ei.shape[1] == 2:
                src, dst = ei[:, 0], ei[:, 1]
            else:
                src, dst = np.array([]), np.array([])

            g: nx.Graph = nx.Graph()
            if len(src) > 0:
                n_nodes = int(max(src.max(), dst.max()) + 1)
                g.add_nodes_from(range(n_nodes))
                g.add_edges_from(zip(src.tolist(), dst.tolist()))
            graphs.append(g)

        from quprep.ingest.graph_ingester import GraphIngester

        ds = GraphIngester(n_features=self.n_graph_features).load(graphs)
        ds.metadata.update(meta_base)
        ds.metadata["edge_index_column"] = self.edge_index_column
        ds.labels = labels
        return ds

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _extract_labels(self, hf_dataset) -> np.ndarray | None:
        """Extract ``target_columns`` from hf_dataset as a labels array."""
        if self.target_columns is None:
            return None
        cols = (
            [self.target_columns]
            if isinstance(self.target_columns, str)
            else list(self.target_columns)
        )
        import pandas as pd

        df = pd.DataFrame({c: hf_dataset[c] for c in cols})
        labels = df.to_numpy()
        return labels.ravel() if labels.shape[1] == 1 else labels

    def _coerce_pil(self, img, PILImage):
        """Ensure *img* is a PIL Image regardless of how HF returned it."""
        if isinstance(img, PILImage.Image):
            return img
        if isinstance(img, dict) and "bytes" in img:
            import io
            return PILImage.open(io.BytesIO(img["bytes"]))
        return PILImage.fromarray(np.asarray(img, dtype=np.uint8))

    def _embed_text(self, texts: list[str]) -> np.ndarray:
        """Return a ``(n_samples, n_features)`` embedding matrix."""
        if self.text_method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(max_features=self.max_features)
            matrix = vec.fit_transform(texts)
            if hasattr(matrix, "toarray"):
                return matrix.toarray().astype(np.float64)
            return np.asarray(matrix, dtype=np.float64)

        if self.text_method == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "text_method='sentence_transformers' requires the "
                    "sentence-transformers package. "
                    "Install it with: pip install quprep[text]"
                ) from e
            model = SentenceTransformer(self.text_model)
            return model.encode(texts, convert_to_numpy=True).astype(np.float64)

        raise ValueError(
            f"text_method must be 'tfidf' or 'sentence_transformers', "
            f"got '{self.text_method}'"
        )
