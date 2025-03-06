# %%
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
import numpy as np
import pprint
import repetition_code_data as rcd
import json
from pathlib import Path
from time import sleep

# Qiskit Runtime
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Qiskit quantum simulator
from qiskit_aer import AerSimulator


# QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR KEY HERE",overwrite = True)
service = QiskitRuntimeService()


run_name = "dist3_time3"
logic_qubits = 1
qubits_per_logical = 3
number_of_measurements = 3
shots = 50000
backend_name = "ibm_kyiv"
version = "0.0"

# Simulate data or run on quantum computer
simulate = True # Warning: be careful to use the correct setting before running on quantum as not to waste service time


backend = service.backend(backend_name)
if simulate:
    backend = AerSimulator.from_backend(backend)
    backend_name += "_simulator"





# Creates a qiskit quantum circuit with k logical qubits, d qubits per logical, and n_measure syndrome measurements
def gen_circuit(k=1, d=3, n_measure = 1):

    # qubits ordered by logical qubit
    qreg_q = [QuantumRegister(2*d-1, f"q{i}") for i in range(k)]

    # Data register ordered by logical qubit
    creg_data = [ClassicalRegister(d, f"data{i}") for i in range(k)]

    # Syndrome register ordered by logical qubit with capacity for multiple measurements
    creg_syndromes = sum([[ClassicalRegister((d-1), f"syndrome_q{i}_m{m}") for m in range(n_measure)] for i in range(k)],[])

    # print(creg_syndromes)
    # print(creg_data)

    circuit = QuantumCircuit(*qreg_q, *creg_data, *creg_syndromes)


    for i in range(k):

        # Entangle redundancy qubits
        for j in range(1,d):
            circuit.cx(qreg_q[i][0], qreg_q[i][j])

        # circuit.barrier(qreg_q[i][0], qreg_q[i][1], qreg_q[i][2], qreg_q[i][3], qreg_q[i][4])
        circuit.barrier([qreg_q[i][j] for j in range(2*d-1)])

        for m in range(n_measure):
            # Stabilizer computation
            for j in range(d-1):
                circuit.cx(qreg_q[i][j], qreg_q[i][d+j])
                circuit.cx(qreg_q[i][j+1], qreg_q[i][d+j])

            # circuit.barrier(qreg_q[i][0], qreg_q[i][1], qreg_q[i][2], qreg_q[i][3], qreg_q[i][4])
            circuit.barrier([qreg_q[i][j] for j in range(2*d-1)])

            # Measure syndrome
            for j in range(d-1):
                circuit.measure(qreg_q[i][j+d], creg_syndromes[i*n_measure+m][j])

        
            circuit.barrier([qreg_q[i][j] for j in range(2*d-1)])

        for j in range(d):
            circuit.measure(qreg_q[i][j], creg_data[i][j])
        # circuit.measure(qreg_q[1], creg_data[1])
        # circuit.measure(qreg_q[2], creg_data[2])

    return circuit


circuit = gen_circuit(logic_qubits,qubits_per_logical,number_of_measurements)
# circuit.draw(output="mpl", style='iqp')



# Optimize circuit
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(circuit)
# isa_circuit.draw('mpl', style='iqp', idle_wires=False)


# Sample data from backend
sampler = Sampler(backend)
job = sampler.run([isa_circuit], shots=shots)
print(job.status())
result = job.result()[0]



# Extract data from result
data = []
for i in range(logic_qubits):
    logic = [eval(f"result.data.data{i}.get_bitstrings()")]
    print(eval(f"result.data.data{i}.get_counts()"))

    for m in range(number_of_measurements):
        d = eval(f"result.data.syndrome_q{i}_m{m}.get_bitstrings()")
        logic.append(d)
    data.append(logic)




# Remove trivial syndromes before saving data
data = np.array(data)
trivial_index = []
for i in range(shots):
    if int("".join(data[0,:,i])) == 0:
        trivial_index.append(i)
data = np.delete(data, trivial_index, axis=2)
non_trivial_shots = data.shape[2]
data = data.tolist()




# Create directories for data

paths = [run_name+"_data", run_name+"_data/Detector_data", run_name+"_data/Error_matrix", run_name+"_data/Format_data", run_name+"_data/Outcome_data", run_name+"_data/Raw_data"]

for p in paths:
    directory_path = Path(p)
    # Create the directory
    try:
        directory_path.mkdir()
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")



# Use repetition_code_data class to structure and save data
settings = [run_name, backend_name, qubits_per_logical, non_trivial_shots, number_of_measurements, version]
data_handler = rcd.repetition_code_data(*settings)


with open(data_handler.run + '_data/Raw_data/result_matrix_'+data_handler.backend_name+'_'+str(data_handler.code_distance)+'_'
    +str(data_handler.shots)+'_'+str(data_handler.time_steps)+'_'+data_handler.version+'.json', 'w') as outfile:
    outfile.write(json.dumps(data[0]))


data_handler.format(raw_data=data[0])


# %%
