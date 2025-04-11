import torch

file_path = r"temp_dir\d3_d_t_1_250225_144033_test_model.pt"


# Load the checkpoint file
checkpoint = torch.load(file_path, map_location="cpu",weights_only=False)

# Extract the model weights (state_dict)
model_weights = checkpoint.get("model", {})

# Print the keys of the model weights
print(model_weights.keys())
