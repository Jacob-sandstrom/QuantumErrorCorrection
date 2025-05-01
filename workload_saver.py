from numpy import pi
import numpy as np
import repetition_code_data as rcd
import json
from pathlib import Path
import string, random

# Qiskit Runtime
from qiskit_ibm_runtime import QiskitRuntimeService




# QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR KEY HERE",overwrite = True)
service = QiskitRuntimeService()


logic_qubits = 1
qubits_per_logical = 17
number_of_measurements = 3
# backend_name = "ibm_kyiv"
backend_name = "ibm_torino"
version = ""
testing_data = False

run_name = f"d{qubits_per_logical}_t{number_of_measurements}_{backend_name.split("_")[-1]}"
if testing_data: run_name += "_testing"


remove_trivial = not testing_data # Removes all trivial syndromes before saving if set to True. Should be False for testing data.



jobs = service.jobs(limit=19, skip=3) # last n jobs
# jobs = service.jobs(limit=20, job_tags=[""]) # last n jobs
# successful_jobs = [j for j in service.jobs(limit=20) if j.status() == "DONE"]

print(jobs)


# workload_id = ""
# retrieved_job = service.job(workload_id)

for retrieved_job in jobs:
    version = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

    result = retrieved_job.result()[0]


    # Extract data from result
    data = []
    for i in range(logic_qubits):
        logic = [eval(f"result.data.data{i}.get_bitstrings()")]
        # print(eval(f"result.data.data{i}.get_counts()"))

        for m in range(number_of_measurements):
            d = eval(f"result.data.syndrome_q{i}_m{m}.get_bitstrings()")
            logic.append(d)
        data.append(logic)

    shots = len(data[0][0])

    # Remove trivial syndromes before saving data
    if remove_trivial:
        data = np.array(data)
        trivial_index = []
        for i in range(shots):
            if int("".join(data[0,:,i])) == 0:
                trivial_index.append(i)
        data = np.delete(data, trivial_index, axis=2)
        saved_shots = data.shape[2]
        data = data.tolist()
    else:
        saved_shots = shots




    # Create directories for data

    paths = ["data/"+run_name, "data/"+run_name+"/Detector_data", "data/"+run_name+"/Error_matrix", "data/"+run_name+"/Format_data", "data/"+run_name+"/Outcome_data", "data/"+run_name+"/Raw_data"]

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
    settings = [run_name, backend_name, qubits_per_logical, saved_shots, number_of_measurements, version]
    data_handler = rcd.repetition_code_data(*settings)


    # with open("data/"+data_handler.run + '/Raw_data/result_matrix_'+data_handler.backend_name+'_'+str(data_handler.code_distance)+'_'
    #     +str(data_handler.shots)+'_'+str(data_handler.time_steps)+'_'+data_handler.version+'.json', 'w') as outfile:
    #     outfile.write(json.dumps(data[0]))


    data_handler.format(raw_data=data[0])
