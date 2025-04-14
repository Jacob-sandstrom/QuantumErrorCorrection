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

###### - DIST 3 TIME 3 -  ########
folder_name = "dist3_time3"
backend_name = "ibm_kyiv"
num_qubits = 3
logic_qubits = 1
qubits_per_logical = 3
n_measurements = 3
n_shots = 3995

print("Dist 3 Time 3:")

for n_shots in [3995, 405748]:
    settings = [folder_name, backend_name, num_qubits, logic_qubits, qubits_per_logical, n_measurements, n_shots]
    print(n_shots," shots :",1-benchmark_mwpm(*settings))

###### - DIST 5 TIME 3 - ########
folder_name = "dist5_time3"
backend_name = "ibm_kyiv"
num_qubits = 5
logic_qubits = 1
qubits_per_logical = 5
n_measurements = 3
n_shots = 3995

print("Dist 3 Time 3:")

for n_shots in [9256, 99606, 99659]:
    settings = [folder_name, backend_name, num_qubits, logic_qubits, qubits_per_logical, n_measurements, n_shots]
    print(n_shots," shots :",1-benchmark_mwpm(*settings))