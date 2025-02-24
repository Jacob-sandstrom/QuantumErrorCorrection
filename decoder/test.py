# %%
import decoder as decoder


d = decoder.Decoder(script_name="test")


# %%

# d.initialise_simulations(0.01)


# %%

d.train()



# %%


# import stim
# error_rate = 0.01
# c = stim.Circuit.generated(
#     # "surface_code:rotated_memory_z",
#     "repetition_code:memory",
#     rounds = 2,
#     distance = 3,
#     after_clifford_depolarization = error_rate,
#     after_reset_flip_probability = error_rate,
#     before_measure_flip_probability = error_rate,
#     before_round_data_depolarization = error_rate)

# s = c.compile_detector_sampler()
# print(c.diagram(type="timeline-svg"))
# s.sample(shots=1)

# %%




# %%

import numpy as np


syndrome = np.array([np.array([np.array([1,0,0,1]),np.array([1,1,1,0])])])

defect_inds = np.nonzero(syndrome)
node_features = np.transpose(np.array(defect_inds)).astype(np.float32)
batch_labels = node_features[:, 0].astype(np.int64)
print("")
# %%
