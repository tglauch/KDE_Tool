#!/usr/bin/env python 

from subprocess import Popen
import numpy as np
import os
import argparse
import itertools
import sys
from time import sleep
from collections import OrderedDict


def parseArguments():
    """Parse the command line arguments
    Returns:
    args : Dictionary containing the command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_bins", type=int, default=30)
    parser.add_argument(
        "--kfold", type=int, default=5)
    parser.add_argument(
        "--gamma", type=float, default=2.0)
    parser.add_argument(
        "--weights", type=str, default='pl')
    parser.add_argument(
        "--model", type=str, required=True)
    parser.add_argument(
        "--outfolder", type=str, default='./out')
    parser.add_argument(
        "--mode", type=str, default='local')
    args = parser.parse_args()
    return args.__dict__


slurm_draft = '#!/usr/bin/env bash \n\
#SBATCH --time=2:30:00 \n\
#SBATCH --mem=8000 \n\
#SBATCH --partition=kta \n\
#SBATCH --error={bpath}/slurm.err \n\
#SBATCH --output={bpath}/slurm.out \n\
bash ./env.sh {args}\n'

args = parseArguments()
print('Run with args')
print(args)
outfolder = os.path.abspath(args['outfolder'])
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
size = 5
bw_dict = OrderedDict([
           ('logsigma', np.linspace(0.06, 0.11, size)),
           ('sinDec' , np.linspace(0.05, 0.3, size)),
          # ('logPsi', np.linspace(0.065, 0.115, size)),
           ('logEr' , np.linspace(0.06, 0.11, size)),
           ])
bins = [bw_dict[key] for key in bw_dict.keys()]
bw_keys = ' '.join(bw_dict.keys())
for i in itertools.product(*bins):
    bw_str = ' '.join([str(a) for a in i])
    ex_args = '--model {} --eval_bins {} --bw_key {} --bw {} --outfolder {} --rs 2 --weights {} --gamma {} --kfold {}'
    ex_args = ex_args.format(args['model'], args['eval_bins'], bw_keys, bw_str, outfolder,
                             args['weights'], args['gamma'], args['kfold'])    
    if args['mode'] == 'local':
        ex_str = 'bash ./env.sh ' + ex_args
        print ex_str
        process = Popen(ex_str, shell=True)
    elif args['mode'] == 'slurm':
        print('Submit over slurm {}'.format(ex_args))
        ex_slurm = slurm_draft.format(bpath=outfolder, args=ex_args)
        with open('temp_submit.sub', "wc") as file:
            file.write(ex_slurm)
        os.system("sbatch {}".format('temp_submit.sub'))
        sleep(0.0001) 
    else:
        print('No valid mode given: please choose between slurm and local')
        sys.exit()
