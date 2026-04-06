"""Image file ingestion — loads images and flattens them into feature vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quprep.core.dataset import Dataset

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class ImageIngester:
    """
    Ingest image files into a Dataset of flattened pixel vectors.

    Loads single images or entire directories. When the directory contains
    subdirectories, the subdirectory name is used as the class label
    (ImageFolder convention). Pixel values are optionally normalized to
    ``[0, 1]`` and resized to a common shape before flattening.

    Requires ``pip install quprep[image]`` (Pillow).

    Parameters
    ----------
    size : tuple of (int, int) or None
        ``(height, width)`` to resize each image to before flattening.
        If ``None``, images are used at their original resolution — all
        images in a batch must then be the same size.
    grayscale : bool
        If ``True`` (default), convert images to grayscale (1 channel).
        If ``False``, keep RGB (3 channels).
    normalize : bool
        If ``True`` (default), divide pixel values by 255 to map to
        ``[0.0, 1.0]``. Set to ``False`` to keep raw ``[0, 255]`` integers.

    Examples
    --------
    Single image::

        ingester = ImageIngester(size=(28, 28))
        dataset = ingester.load("cat.png")

    Directory with class labels::

        # images/cat/img1.jpg, images/dog/img1.jpg
        ingester = ImageIngester(size=(32, 32))
        dataset = ingester.load("images/")
        print(dataset.labels)        # ['cat', 'cat', ..., 'dog', ...]
        print(dataset.data.shape)    # (n_images, 32*32)
    """

    def __init__(
        self,
        size: tuple[int, int] | None = (28, 28),
        grayscale: bool = True,
        normalize: bool = True,
    ):
        self.size = size
        self.grayscale = grayscale
        self.normalize = normalize

    def load(self, source: str | Path) -> Dataset:
        """
        Load one or more images and return a Dataset.

        Parameters
        ----------
        source : str or Path
            Path to a single image file or to a directory of images.
            Supported formats: PNG, JPG/JPEG, BMP, TIFF, WebP.

            *Directory loading* — two layouts are supported:

            - **Flat**: all image files at the top level → no labels.
            - **Subfolders**: each subdirectory is a class; images inside
              are samples → ``dataset.labels`` holds the class name strings.

        Returns
        -------
        Dataset
            ``data`` shape is ``(n_samples, n_pixels)`` where
            ``n_pixels = height × width`` (grayscale) or
            ``height × width × 3`` (RGB).
            ``metadata["modality"]`` is ``"image"``.
            ``metadata["size"]`` is the ``(H, W)`` tuple used.
            ``metadata["channels"]`` is ``1`` (grayscale) or ``3`` (RGB).

        Raises
        ------
        ImportError
            If Pillow is not installed.
        FileNotFoundError
            If ``source`` does not exist.
        ValueError
            If no supported image files are found, or images have
            mismatched sizes when ``size=None``.
        """
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "ImageIngester requires Pillow. Install it with: pip install quprep[image]"
            ) from e

        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Path not found: {source}")

        if source.is_file():
            arr = self._load_one(source, Image)
            n_pixels = arr.shape[0]
            channels = 1 if self.grayscale else 3
            h, w = self.size if self.size else (n_pixels // channels, -1)
            return Dataset(
                data=arr.reshape(1, -1),
                feature_names=[f"px_{i}" for i in range(arr.shape[0])],
                feature_types=["continuous"] * arr.shape[0],
                metadata={
                    "source": str(source),
                    "modality": "image",
                    "size": self.size,
                    "channels": 1 if self.grayscale else 3,
                },
            )

        # --- directory ---
        paths, labels = self._collect(source)

        arrays = [self._load_one(p, Image) for p in paths]
        self._check_shapes(arrays, paths)

        data = np.stack(arrays, axis=0)          # (n, n_pixels)
        n_pixels = data.shape[1]
        label_arr = np.array(labels) if any(l is not None for l in labels) else None

        return Dataset(
            data=data,
            feature_names=[f"px_{i}" for i in range(n_pixels)],
            feature_types=["continuous"] * n_pixels,
            metadata={
                "source": str(source),
                "modality": "image",
                "size": self.size,
                "channels": 1 if self.grayscale else 3,
                "n_images": len(paths),
            },
            labels=label_arr,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_one(self, path: Path, Image) -> np.ndarray:
        """Load, convert, resize, and flatten a single image."""
        img = Image.open(path)
        img = img.convert("L" if self.grayscale else "RGB")
        if self.size is not None:
            # PIL uses (width, height) — reverse our (H, W) convention
            img = img.resize((self.size[1], self.size[0]))
        arr = np.asarray(img, dtype=float)
        if self.normalize:
            arr = arr / 255.0
        return arr.flatten()

    def _collect(self, directory: Path) -> tuple[list[Path], list[str | None]]:
        """Return (image_paths, labels). Labels are None when no subdirs."""
        subdirs = [d for d in sorted(directory.iterdir()) if d.is_dir()]

        if subdirs:
            paths, labels = [], []
            for subdir in subdirs:
                class_name = subdir.name
                for p in sorted(subdir.iterdir()):
                    if p.suffix.lower() in _IMAGE_EXTENSIONS:
                        paths.append(p)
                        labels.append(class_name)
        else:
            paths = sorted(
                p for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            )
            labels = [None] * len(paths)

        if not paths:
            raise ValueError(
                f"No supported image files found in '{directory}'. "
                f"Supported extensions: {sorted(_IMAGE_EXTENSIONS)}"
            )
        return paths, labels

    def _check_shapes(self, arrays: list[np.ndarray], paths: list[Path]) -> None:
        """Raise if images have different flattened sizes (only when size=None)."""
        if self.size is not None:
            return
        sizes = {a.shape[0] for a in arrays}
        if len(sizes) > 1:
            raise ValueError(
                "Images have different sizes. Set size=(H, W) to resize them "
                "to a common shape before flattening."
            )
