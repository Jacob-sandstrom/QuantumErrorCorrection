### Code for benchmarking MWPM performance.
import numpy as np

import repetition_code_data as rcd
import repetition_code_MWPM as rcdmwpm
import json

def benchmark_mwpm(folder_name, backend_name, num_qubits, logic_qubits, qubits_per_logical, n_measurements, n_shots):

    settings = ["data/"+folder_name, backend_name, qubits_per_logical, n_shots, n_measurements, "0.0"]

    # Create a rcdmpwm-instance with the given settings.
    mwpm = rcdmwpm.repetition_code_MWPM(*settings)

    # Run the MWPM-algorithm and print the error-rate
    return(mwpm.MWPM(weight="1"))

# Testing data on Torino

for i in range(3,9,2):
    folder_name = f"d{i}_t3_torino_testing"
    backend_name = "ibm_torino"
    num_qubits = i
    logic_qubits = 1
    qubits_per_logical = i
    n_measurements = 3
    n_shots = 500000

    print(f"Dist {i} Time 3:")

    settings = [folder_name, backend_name, num_qubits, logic_qubits, qubits_per_logical, n_measurements, n_shots]
    print(n_shots," shots :",1-benchmark_mwpm(*settings))