import json
import numpy as np

file = "data/d7_t3_torino/Outcome_data/outcome_dict_ibm_torino_7_1781686_3_0.0.json"

splitfraction = 2

with open(file, 'r') as infile:
    data = json.load(infile)


print(data["0"])

# d1 = dict(data.items()[len(data)/2:])

# print(dict(list(data.items())[:2]))

d1 = dict(list(data.items())[:(len(data)//splitfraction)+3])
d2 = dict(list(data.items())[(len(data)//splitfraction)+3:])



d2 = { str(int(k)-len(d1)): v for k, v in d2.items() }
# print(dict(list(d2.items())[:2]))

fparts = file.split("/")[:-1]
ofile1 = fparts[0] + "/" + fparts[1] + "/" + fparts[2] + "/" + str(len(d1)) + ".json"
ofile2 = fparts[0] + "/" + fparts[1] + "/" + fparts[2] + "/" + str(len(d2)) + ".json"


ofile1 = "_".join(map(lambda x: x, file.split("_")[:-3])) + "_" + str(len(d1)) + "_" + "_".join(map(lambda x: x, file.split("_")[-2:]))
ofile2 = "_".join(map(lambda x: x, file.split("_")[:-3])) + "_" + str(len(d2)) + "_" + "_".join(map(lambda x: x, file.split("_")[-2:]))

print(ofile1)
print(ofile2)


# print("_".join(map(lambda x: x, file.split("_")[:-3])) + "_" + str(len(d1)) + "_" + "_".join(map(lambda x: x, file.split("_")[-2:])))

with open(ofile1, 'w') as outfile:
    outfile.write(json.dumps(d1))

with open(ofile2, 'w') as outfile:
    outfile.write(json.dumps(d2))