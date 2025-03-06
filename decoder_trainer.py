
import json
import numpy as np
import torch


logic_qubits = 1
qubits_per_logical = 3
number_of_measurements = 3
# shots = 100000
shots = 10
# backend_name = "ibm_brussels"
# backend_name = "ibm_kyiv_simulator"
backend_name = "ibm_kyiv"
run = "test"
version = "0.0"
code_distance = qubits_per_logical
time_steps = number_of_measurements

settings = [run, backend_name, qubits_per_logical, shots, number_of_measurements, "0.0"]


outcome_file = run + '_data/Outcome_data/outcome_dict_'+backend_name+'_'+str(code_distance)+'_'+str(shots)+'_'+str(time_steps)+'_'+version+'.json'
syndromes_file = run + '_data/Detector_data/detector_dict_'+backend_name+'_'+str(code_distance)+'_'+str(shots)+'_'+str(time_steps)+'_'+version+'.json'



from decoder import decoder as decoder


d = decoder.Decoder(script_name="test")


# %%

# d.initialise_simulations(0.01)


# %%

d.train()


