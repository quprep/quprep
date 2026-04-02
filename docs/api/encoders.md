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

## EntangledAngleEncoder

::: quprep.encode.entangled_angle.EntangledAngleEncoder
    options:
      show_source: true

---

## IQPEncoder

::: quprep.encode.iqp.IQPEncoder
    options:
      show_source: true

---

## ReUploadEncoder

::: quprep.encode.reupload.ReUploadEncoder
    options:
      show_source: true

---

## HamiltonianEncoder

::: quprep.encode.hamiltonian.HamiltonianEncoder
    options:
      show_source: true

---

## ZZFeatureMapEncoder

::: quprep.encode.zz_feature_map.ZZFeatureMapEncoder
    options:
      show_source: true

---

## PauliFeatureMapEncoder

::: quprep.encode.pauli_feature_map.PauliFeatureMapEncoder
    options:
      show_source: true

---

## RandomFourierEncoder

::: quprep.encode.random_fourier.RandomFourierEncoder
    options:
      show_source: true

---

## TensorProductEncoder

::: quprep.encode.tensor_product.TensorProductEncoder
    options:
      show_source: true

---

## QAOAProblemEncoder

::: quprep.encode.qaoa_problem.QAOAProblemEncoder
    options:
      show_source: true
