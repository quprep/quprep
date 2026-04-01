# QuPrep

**The missing preprocessing layer between classical datasets and quantum computing frameworks.**

```
CSV / DataFrame / NumPy  →  QuPrep  →  circuit-ready output
```

QuPrep converts classical datasets into quantum-circuit-ready format. It is not a quantum computing framework, simulator, or training tool — it is the preprocessing step that feeds into Qiskit, PennyLane, Cirq, TKET, and any other quantum workflow.

<div class="grid cards" markdown>

-   :fontawesome-solid-download:{ .lg .middle } **Install in seconds**

    ---

    ```bash
    pip install quprep
    ```

    No quantum framework required for the core install.

    [:octicons-arrow-right-24: Installation guide](getting-started/installation.md)

-   :fontawesome-solid-rocket:{ .lg .middle } **Zero to circuit in one line**

    ---

    ```python
    import quprep
    result = quprep.prepare("data.csv", encoding="angle")
    print(result.circuit)
    ```

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :fontawesome-solid-lightbulb:{ .lg .middle } **Not sure which encoding?**

    ---

    ```python
    rec = quprep.recommend("data.csv", task="classification", qubits=8)
    result = rec.apply("data.csv")
    ```

    [:octicons-arrow-right-24: Encoding guide](guides/encodings.md)

-   :fontawesome-solid-cube:{ .lg .middle } **QUBO & quantum optimization**

    ---

    ```python
    from quprep.qubo import max_cut, solve_brute, qaoa_circuit
    q = max_cut(adj)
    qasm = qaoa_circuit(q, p=2)
    ```

    [:octicons-arrow-right-24: QUBO guide](guides/qubo.md)

</div>

---

## Pipeline stages

| Stage | Since | Description |
|---|---|---|
| **Ingest** | v0.1.0 | CSV, TSV, NumPy arrays, Pandas DataFrames |
| **Clean** | v0.1.0 | Missing values, outliers, categoricals, feature selection |
| **Normalize** | v0.1.0 | Auto-selected per encoding (L2, MinMax, Z-score, binary) |
| **Encode** | v0.1.0 | Angle, Amplitude, Basis |
| **Export** | v0.1.0 | OpenQASM 3.0, Qiskit |
| **Reduce** | v0.2.0 | PCA, LDA, DFT, t-SNE, UMAP, hardware-aware |
| **Encode+** | v0.2.0 | IQP, Entangled Angle, Data re-uploading, Hamiltonian |
| **Export+** | v0.2.0 | PennyLane, Cirq, TKET, ASCII + matplotlib visualization |
| **Recommend** | v0.2.0 | Automatic encoding selection for your dataset and task |
| **QUBO** | v0.3.0 | QUBO/Ising, 7 problem formulations, solvers, QAOA, D-Wave export |
| **Validate** | v0.4.0 | Input validation, schema enforcement, cost estimation, sklearn fit/transform, `import quprep as qd` |
| **Intelligence** | v0.5.0 | Qubit suggestion, encoding comparison, data drift detection, pipeline save/load, batch QASM export |
| **Encode++** | v0.6.0 | ZZFeatureMap, PauliFeatureMap, RandomFourier, TensorProduct encoders |
| **Export++** | v0.6.0 | Amazon Braket, Q# (Azure Quantum), IQM native format |
| **Plugins** | v0.6.0 | `register_encoder` / `register_exporter` — custom encoders/exporters via `prepare()` |

---

## Supported frameworks

| Framework | Install | Output type |
|---|---|---|
| OpenQASM 3.0 | *(no extra deps)* | `str` |
| Qiskit | `quprep[qiskit]` | `QuantumCircuit` |
| PennyLane | `quprep[pennylane]` | `qml.QNode` |
| Cirq | `quprep[cirq]` | `cirq.Circuit` |
| TKET | `quprep[tket]` | `pytket.Circuit` |
| Amazon Braket | `quprep[braket]` | `braket.circuits.Circuit` |
| Q# / Azure Quantum | `quprep[qsharp]` | `str` (Q# 1.0 source) |
| IQM | `quprep[iqm]` | `dict` (PRX+CZ JSON) |
| D-Wave Ocean | *(via `.to_dwave()`)* | BQM dict |

---

## What QuPrep does NOT do

QuPrep is intentionally narrow in scope. It does not:

- Train quantum machine learning models
- Simulate quantum circuits
- Execute on quantum hardware
- Optimize variational parameters
- Replace Qiskit, PennyLane, Cirq, or any other framework

It prepares your data. Everything else is your framework's job.

---

## CLI

```bash
# Encode a CSV to OpenQASM 3.0
quprep convert data.csv --encoding angle

# Get an encoding recommendation
quprep recommend data.csv --task classification --qubits 8

# QUBO problems
quprep qubo maxcut --adjacency "0,1,1;1,0,1;1,1,0" --solve
quprep qubo qaoa maxcut --adjacency "0,1,1;1,0,1;1,1,0" --p 2 --output circuit.qasm
```
