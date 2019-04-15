import numpy as np
import os
import argparse

def parseArguments():
    """Parse the command line arguments
    Returns:
    args : Dictionary containing the command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str, required=True)
    args = parser.parse_args()
    return args.__dict__

args = parseArguments() 
cv_folder = args['folder']

files = [i for i in sorted(os.listdir(cv_folder)) if '.npy' in i]

res_dict = {'mean_llh': [],'var_llh': [], 'zeros': []}
fsplit = files[0].split('.npy')[0].replace('_r', 'r').split('_')
keys = fsplit[0:len(fsplit):2]
for i in keys:
    res_dict[i] = []

for f in files:
    fsplit = f.split('.npy')[0].replace('_r', 'r').split('_')
    values = fsplit[1:len(fsplit):2]
    x = np.load(os.path.join(cv_folder, f))[()]
    for i,key in enumerate(keys):
        res_dict[key].append(float(values[i]))
    res_dict['mean_llh'].append(np.mean(x['lh'])*-1)
    res_dict['var_llh'].append(np.std(x['lh']))
    res_dict['zeros'].append(np.sum(x['zeros']))
    
keys.extend(['mean_llh', 'var_llh', 'zeros'])
inds = np.argsort(res_dict['mean_llh'])

print('Sorted by mean')
for i in inds[:200]:
    pstr = ''
    for key in keys:
        pstr += '{} {:.3f} '.format(key, res_dict[key][i])
    print pstr


print('Max value')
i = inds[-1]
pstr = ''
for key in keys:
    pstr += '{} {:.3f} '.format(key, res_dict[key][i])
print pstr
