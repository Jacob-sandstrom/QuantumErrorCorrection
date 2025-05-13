import json
from collections import Counter
import os
import matplotlib.pyplot as plt
import numpy as np
# Loop over each file in the directory and read JSON


dir_paths = [f"data/d{x}_t3_torino_testing/Outcome_data" for x in range(3,24,2)]
print("Directory paths:", dir_paths)


all_err_counts = []
all_err_freqs = []

for directory_path in dir_paths:
    file_names = os.listdir(directory_path)
    # print("Files in directory:", file_names)


    errors = 0
    n_data = 0
    for file_name in file_names:
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        for d in data.values():
            n_data += 1
            if d[0] == 1:
                errors += 1
        

    print("Directory:", directory_path)
    print("Errors:", errors)
    print("Error frequency", errors/n_data)
    all_err_counts.append(errors)
    all_err_freqs.append(errors/n_data)

# Plot correct frequencies
plt.figure(figsize=(10, 6))
x_values = range(3, 24, 2)
correct_frequencies = [1 - freq for freq in all_err_freqs]
plt.plot(x_values, correct_frequencies, marker='o', linestyle='-', color='g', label='Rättfrekvens')
plt.title('Rättfrekvens mot Koddistans')
plt.xlabel('Koddistans')
plt.ylabel('Rättfrekvens')
plt.grid(True)

plt.ylim(0, 1)
plt.xticks(x_values, [str(x) for x in x_values])
plt.legend()
plt.show()

print(correct_frequencies)