"""Text ingestion — converts raw text to dense feature vectors for quantum encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quprep.core.dataset import Dataset

_VALID_METHODS = ("tfidf", "sentence_transformers")


class TextIngester:
    """
    Convert text data into a Dataset of dense feature vectors.

    Two embedding methods are supported:

    - **tfidf** *(default, no extra deps)* — TF-IDF bag-of-words via
      sklearn. Sparse output is converted to dense. Use a
      :class:`~quprep.reduce.pca.PCAReducer` afterwards to bring the
      feature count down to a circuit-friendly size.
    - **sentence_transformers** — semantic sentence embeddings via the
      ``sentence-transformers`` library. Produces compact (384–768d)
      dense vectors that are directly suitable for angle or amplitude
      encoding. Requires ``pip install quprep[text]``.

    Parameters
    ----------
    method : str
        ``'tfidf'`` (default) or ``'sentence_transformers'``.
    model : str
        Sentence-transformers model name. Only used when
        ``method='sentence_transformers'``.
        Default: ``'all-MiniLM-L6-v2'`` (384-d, fast, good quality).
    max_features : int or None
        Maximum vocabulary size for TF-IDF. Ignored for
        sentence_transformers. Default 512.
    text_column : str or None
        Column name containing text when loading a CSV file. Required
        for CSV sources; ignored for ``.txt`` files and list inputs.
    target_column : str or list of str or None
        Column name(s) to treat as labels. Stored in ``Dataset.labels``.
    delimiter : str
        CSV delimiter. Default ``','``.

    Examples
    --------
    From a list of strings::

        ingester = TextIngester(method="tfidf", max_features=64)
        dataset = ingester.load(["quantum is great", "machine learning rocks"])

    From a text file (one sentence per line)::

        dataset = TextIngester().load("corpus.txt")

    From a CSV::

        ingester = TextIngester(text_column="review", target_column="sentiment")
        dataset = ingester.load("reviews.csv")
        print(dataset.labels)   # sentiment column values

    With sentence transformers::

        ingester = TextIngester(method="sentence_transformers")
        dataset = ingester.load(sentences)
        # dataset.data.shape → (n, 384) — directly encode with AngleEncoder
    """

    def __init__(
        self,
        method: str = "tfidf",
        model: str = "all-MiniLM-L6-v2",
        max_features: int | None = 512,
        text_column: str | None = None,
        target_column: str | list[str] | None = None,
        delimiter: str = ",",
    ):
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got '{method}'")
        self.method = method
        self.model = model
        self.max_features = max_features
        self.text_column = text_column
        self.target_column = target_column
        self.delimiter = delimiter

    def load(self, source) -> Dataset:
        """
        Load text data and return a Dataset of feature vectors.

        Parameters
        ----------
        source : list of str, str, or Path
            - **list of str** — texts are used directly.
            - **.txt file** — each non-empty line is one text sample.
            - **.csv file** — ``text_column`` must be set; rows become samples.

        Returns
        -------
        Dataset
            ``data`` shape is ``(n_samples, n_features)`` where
            ``n_features`` is ``max_features`` (TF-IDF) or the embedding
            dimension (sentence_transformers).
            ``metadata["modality"]`` is ``"text"``.
            ``metadata["method"]`` is the embedding method used.

        Raises
        ------
        ImportError
            If ``method='sentence_transformers'`` and the package is not
            installed.
        FileNotFoundError
            If a file path is provided but does not exist.
        ValueError
            If a CSV is passed but ``text_column`` is not set, or the
            column is not found.
        """
        texts, labels = self._load_texts(source)
        vectors = self._embed(texts)

        n_features = vectors.shape[1]
        return Dataset(
            data=vectors,
            feature_names=[f"emb_{i}" for i in range(n_features)],
            feature_types=["continuous"] * n_features,
            metadata={
                "modality": "text",
                "method": self.method,
                "n_samples": len(texts),
                "n_features": n_features,
            },
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_texts(self, source) -> tuple[list[str], np.ndarray | None]:
        """Return (texts, labels). Labels are None when no target_column."""
        if isinstance(source, (list, tuple)):
            return list(source), None

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() == ".txt":
            texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
            return texts, None

        # CSV path
        import pandas as pd

        df = pd.read_csv(path, delimiter=self.delimiter)

        if self.text_column is None:
            raise ValueError(
                "text_column must be set when loading a CSV file. "
                f"Available columns: {list(df.columns)}"
            )
        if self.text_column not in df.columns:
            raise ValueError(
                f"Column '{self.text_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        texts = df[self.text_column].fillna("").astype(str).tolist()

        labels = None
        if self.target_column is not None:
            cols = (
                [self.target_column]
                if isinstance(self.target_column, str)
                else list(self.target_column)
            )
            label_arr = df[cols].to_numpy()
            labels = label_arr.ravel() if label_arr.shape[1] == 1 else label_arr

        return texts, labels

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Convert texts to a dense (n_samples, n_features) matrix."""
        if self.method == "tfidf":
            return self._embed_tfidf(texts)
        return self._embed_st(texts)

    def _embed_tfidf(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(max_features=self.max_features)
        matrix = vec.fit_transform(texts)
        # always return dense float64
        if hasattr(matrix, "toarray"):
            return matrix.toarray().astype(np.float64)
        return np.asarray(matrix, dtype=np.float64)

    def _embed_st(self, texts: list[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence_transformers method requires the sentence-transformers package. "
                "Install it with: pip install quprep[text]"
            ) from e
        model = SentenceTransformer(self.model)
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float64)
