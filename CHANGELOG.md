# Changelog

All notable changes to QuPrep will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
QuPrep uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Project scaffold and package structure
- `pyproject.toml` with optional framework dependencies (`qiskit`, `pennylane`, `cirq`, `tket`, `all`)
- GitHub Actions CI pipeline
- MkDocs documentation setup

---

## [0.1.0] — _not yet released_

> First public alpha. Target: Phase 1 (Weeks 1–4).

### Planned
- Data ingestion: CSV, NumPy, Pandas with auto type detection
- Basic cleaning: imputation, outlier handling
- Normalization: Min-Max, Z-score, L2 with auto-selection per encoding
- Core encoders: Angle (Ry), Amplitude, Basis
- Exporters: Qiskit, OpenQASM 3.0
- CLI: `quprep convert`
- PyPI release

---

[Unreleased]: https://github.com/quprep/quprep/compare/HEAD...HEAD
