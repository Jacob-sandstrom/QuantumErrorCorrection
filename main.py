
import json
import numpy as np

logic_qubits = 1
qubits_per_logical = 3
number_of_measurements = 3
shots = 10
backend_name = "ibm_brussels"
run = "test"
version = "0.0"
code_distance = qubits_per_logical
time_steps = number_of_measurements

settings = [run, backend_name, qubits_per_logical, shots, number_of_measurements, "0.0"]


with open(run + '_data/Format_data/result_dict_'+backend_name+'_'+str(code_distance) 
          +'_'+str(shots)+'_'+str(time_steps)+'_'+version+'.json', 'r') as infile:
    data = json.load(infile)


d0 = data["0"]
n_shots = len(data)
n_measure = len(d0)
n_x = len(d0[0])
# print(d0)

# syndromes in the shape [n_shots, x_coordinate, z_coordinate, time]
# z_coordinate is always 1 for repetition code
shots = np.empty((n_shots,n_x,1,n_measure))
for i, shot in enumerate(data.keys()):
    d = np.transpose(np.array(data[shot]))
    d = np.reshape(d, (n_x,1,n_measure))
    shots[i] = d

# print("\n\n")
print(shots[0])
print(shots.shape)

from decoder import graph_representation as gr

node_features, batch_labels = gr.get_batch_of_node_features(shots)

print(node_features)
print(batch_labels)
# gr.get_batch_of_edges(node_features, batch_labels, device)