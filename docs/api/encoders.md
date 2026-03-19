# Encoders

Encoders map a normalized feature vector to a parameterized quantum circuit.

All encoders:

- Accept a 1-D `np.ndarray`
- Return an `EncodedResult` with `.parameters` and `.metadata`
- Are deterministic — same input always produces same output
- Do **not** normalize — normalization is the pipeline's job

---

## AngleEncoder

::: quprep.encode.angle.AngleEncoder
    options:
      show_source: true

---

## AmplitudeEncoder

::: quprep.encode.amplitude.AmplitudeEncoder
    options:
      show_source: true

---

## BasisEncoder

::: quprep.encode.basis.BasisEncoder
    options:
      show_source: true

---

## BaseEncoder and EncodedResult

::: quprep.encode.base.BaseEncoder
    options:
      show_source: false

::: quprep.encode.base.EncodedResult
    options:
      show_source: false

---

## Coming in v0.2.0

| Encoder | Module | Reference |
|---|---|---|
| `IQPEncoder` | `quprep.encode.iqp` | Havlíček et al., Nature 2019 |
| `ReUploadEncoder` | `quprep.encode.reupload` | Pérez-Salinas et al., Quantum 2020 |
| `HamiltonianEncoder` | `quprep.encode.hamiltonian` | — |
