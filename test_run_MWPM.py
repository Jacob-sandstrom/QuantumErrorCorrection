### Code for benchmarking MWPM performance.
import numpy as np

import repetition_code_data as rcd
import repetition_code_MWPM as rcdmwpm
import json

# Specify which run of data to use
backend_name = "ibm_kyiv_simulator"
num_qubits = 5
logic_qubits = 1
qubits_per_logical = 3
number_of_measurements = 3
n_syndrome_shots = 1000000

settings = ["test", backend_name, qubits_per_logical, n_syndrome_shots, number_of_measurements, "0.0"]

# Create a rcdmpwm-instance with the given settings.
mwpm = rcdmwpm.repetition_code_MWPM(*settings)

# Run the MWPM-algorithm and print the error-rate
print(mwpm.MWPM(weight="1"))