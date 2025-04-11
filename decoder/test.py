import decoder as decoder
import torch
from torch.nn.functional import sigmoid
from graph_representation import fetch_data, get_batch_of_node_features
import numpy as np

# Ladda in tränad model
d = decoder.Decoder(script_name="test")
d.load_trained_model()  # Ändra modell i utils-saved_model_path

d.model.eval()

# Data att testa på
outcome_file = r"dist3_time3_data\Outcome_data\outcome_dict_ibm_kyiv_simulator_3_70286_3_0.0.json"
syndromes_file = r"dist3_time3_data\Detector_data\detector_dict_ibm_kyiv_simulator_3_70286_3_0.0.json"

all_syndromes, all_flips, all_trivial = fetch_data(outcome_file, syndromes_file, d.device)

# Filtrerar ut triviala exempel (egentiligen redan gjort)
non_trivial_mask = np.logical_not(all_trivial)
filtered_syndromes = all_syndromes[non_trivial_mask]
filtered_flips = all_flips[non_trivial_mask]

syndromes = filtered_syndromes[:]  
flips = filtered_flips[:]

x, edge_index, batch_labels, edge_attr = d.get_batch_of_graphs(syndromes)

# Normalisering från decoder
x[:, 1] = (x[:, 1] - (d.d_t / 2)) / (d.d_t / 2)
x[:, 2:] = (x[:, 2:] - (d.code_size / 2)) / (d.code_size / 2)

guesses0 = 0 # Räknar upp hur många nollor vi får
correct_preds = 0 # Räknar upp hur många som blir rätt

#Nätverket räknar
with torch.no_grad():
    out = d.model(x, edge_index, batch_labels, edge_attr)
    prediction = (d.sigmoid(out.detach()) > 0.5).long()
    target = flips.long()
    guesses0+= int(((prediction == 0).sum(dim=1) == 
                                  d.model_settings["num_classes"]).sum().item()) #Ifall guess = 0
    correct_preds += int(((prediction == target).sum(dim=1) == 
                                  d.model_settings["num_classes"]).sum().item()) #Ifall prediction = target
    
    
print("Prediction:", prediction)
print("Answer/truth:", flips)
print("Andel noll gissningar:",(guesses0)/len(prediction)) 

print("Andel rätta gissningar:",(correct_preds)/d.training_settings["dataset_size"])
