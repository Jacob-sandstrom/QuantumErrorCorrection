### Code for benchmarking MWPM performance.
import numpy as np
import math

import repetition_code_data as rcd
import repetition_code_MWPM as rcdmwpm
import json

#from graph_representation import fetch_data

def benchmark_mwpm(folder_name, backend_name, num_qubits, logic_qubits, qubits_per_logical, n_measurements, n_shots, version, ignoreTrivial):

    settings = ["data/"+folder_name, backend_name, qubits_per_logical, n_shots, n_measurements, version]

    # Create a rcdmpwm-instance with the given settings.
    mwpm = rcdmwpm.repetition_code_MWPM(*settings)

    # Run the MWPM-algorithm and print the error-rate
    return(mwpm.MWPM(weight="1",ignoreTrivial=ignoreTrivial))

# Testing data on Torino

for i in [3,5,7,9,11]:


    folder_name = f"d{i}_t3_torino_testing"
    backend_name = "ibm_torino"
    num_qubits = i
    logic_qubits = 1
    qubits_per_logical = i
    n_measurements = 3
    n_shots = 500000

    print(f"#### Dist {i} Time 3: ####")

    print("With trivial data:")
    settings = [folder_name, backend_name, num_qubits, logic_qubits, qubits_per_logical, n_measurements, n_shots, "0.0"]
    print("accuracy:",1-benchmark_mwpm(*settings, False))
    
    print("- \nNon-trivial data:")
    settings = [folder_name, backend_name, num_qubits, logic_qubits, qubits_per_logical, n_measurements, n_shots, "0.0"]
    print("accuracy:",1-benchmark_mwpm(*settings, True))

    print()

#Data is split up for higher code distances than 11

#versionDict contains the "version" labels for all different data partitions for each code distance
versionDict = {13: ["11JJ", "ZD8Q"]}

for i in [13]:


    folder_name = f"d{i}_t3_torino_testing"
    backend_name = "ibm_torino"
    num_qubits = i
    logic_qubits = 1
    qubits_per_logical = i
    n_measurements = 3
    n_shots = 250000

    #The version labels for the current code distance
    versions = versionDict[i] 
    numVersions = len(versions)

    print(f"#### Dist {i} Time 3 ####")

    print("With trivial data:")
    accuracy = 0
    for ver in versions:
        settings = [folder_name, backend_name, num_qubits, logic_qubits, qubits_per_logical, n_measurements, n_shots, ver]
        accuracy += 1-benchmark_mwpm(*settings, False)
    accuracy /= numVersions
    print("accuracy:",accuracy)
    
    print("- \nNon-trivial data:")
    accuracy = 0
    for ver in versions:
        settings = [folder_name, backend_name, num_qubits, logic_qubits, qubits_per_logical, n_measurements, n_shots, ver]
        accuracy += 1-benchmark_mwpm(*settings, True)
    accuracy /= numVersions
    print("accuracy:",accuracy)

    print()