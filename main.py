
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


# with open(run + '_data/Detector_data/detector_dict_'+backend_name+'_'+str(code_distance) 
#           +'_'+str(shots)+'_'+str(time_steps)+'_'+version+'.json', 'r') as infile:
#     syndrome_data = json.load(infile)

# with open(run + '_data/Outcome_data/outcome_dict_'+backend_name+'_'+str(code_distance) 
#           +'_'+str(shots)+'_'+str(time_steps)+'_'+version+'.json', 'r') as infile:
#     outcome_data = json.load(infile)


outcome_file = run + '_data/Outcome_data/outcome_dict_'+backend_name+'_'+str(code_distance)+'_'+str(shots)+'_'+str(time_steps)+'_'+version+'.json'
syndromes_file = run + '_data/Detector_data/detector_dict_'+backend_name+'_'+str(code_distance)+'_'+str(shots)+'_'+str(time_steps)+'_'+version+'.json'



# n_outcome_shots = len(outcome_data)
# o0 = outcome_data["0"]


# s0 = syndrome_data["0"]
# n_syndrome_shots = len(syndrome_data)
# n_measure = len(s0)
# n_x = len(s0[0])
# # print(s0)

# if n_outcome_shots != n_syndrome_shots:
#     raise Exception("Number of syndromes in outcome and syndromes mismatch")


# flips = []
# for i, shot in enumerate(outcome_data.keys()):
#     flips.append([int(outcome_data[shot][0])])
# flips = torch.tensor(flips)

# # syndromes in the shape [n_syndrome_shots, x_coordinate, z_coordinate, time]
# # z_coordinate is always 1 for repetition code
# syndromes = np.empty((n_syndrome_shots,n_x,1,n_measure))
# for i, shot in enumerate(syndrome_data.keys()):
#     d = np.transpose(np.array(syndrome_data[shot]))
#     d = np.reshape(d, (n_x,1,n_measure))
#     syndromes[i] = d

# # print("\n\n")
# print(syndromes[0])
# print(syndromes.shape)

from decoder import graph_representation as gr

# node_features, batch_labels = gr.get_batch_of_node_features(syndromes)

# print(node_features)
# print(batch_labels)
# gr.get_batch_of_edges(node_features, batch_labels, device)


syndromes, flips, trivial = gr.fetch_data(outcome_file, syndromes_file)
print(trivial)
print(flips)
print(syndromes[np.logical_not(trivial)].shape)