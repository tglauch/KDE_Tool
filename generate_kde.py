#!/usr/bin/env python

# In order to run this File you need to define a model file
# see the exmples in ./models  and pass this as command line argument.
# Further non required command line arguments are 'gamma' - the power law spectrum -
# for the MC weighting and the number of bind for the evaluation grid

# Example:

# python generate_kde.py --model ./models/example.py --gamma 2.19 --eval_bins 100

import itertools
import importlib
from kde import *
from numpy.lib.recfunctions import append_fields
import cPickle as pickle
import argparse
from collections import OrderedDict
from functions import GreatCircleDistance, powerlaw, load_model, read_config
from mc_cuts import diffuse_cuts as mc_cut
import sys
import os
import time
import numpy


def parseArguments():
    """Parse the command line arguments
    Returns:
    args : Dictionary containing the command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_bins",
        type=int, default=30)
    parser.add_argument(
        "--gamma",
        type=float, default=2.19)
    parser.add_argument(
        "--phi0",
        help="in units of 1e-18 1/GeV/cm^2/sr/s",
        type=float, default=1.01)
    parser.add_argument(
        "--model",
        type=str, required=True)
    parser.add_argument(
        "--mc",
        type=str, required=False)
    parser.add_argument(
        "--weights",
        type=str, default='pl')
    # pl = powerlaw with index in args['gamma'] and normalization in args['phi0']
    # conv = conventional atmospheric
    # prompt = prompt atmopsheric
    parser.add_argument(
        "--save_str",
        type=str, default='')
    parser.add_argument(
        "--outfolder",
        type=str)
    parser.add_argument(
        '--no_save', default=False,
        action='store_true')
    parser.add_argument(
        '--adaptive', default=False,
        action='store_true')
    args = parser.parse_args()
    return args.__dict__


def create_KDE(args, inds=None, bws={}, mc=None):
    if 'mc' not in args.keys():
        args['mc'] = None
    if 'phi0' not in args.keys():
        args['phi0'] = 1
    if args['outfolder'] is None:
        args['outfolder'] = os.path.join(os.path.dirname(args['model']), 'out')
    args['phi0'] *= 1e-18  # correct units
    t0 = time.time()
    model, mname = load_model(args['model'])
    print('---- Run KDE with args:')
    print(args)
    if not os.path.exists(args['outfolder']):
        os.makedirs(args['outfolder'])

    print('Load and Update the Monte Carlo')
    config = read_config()
    cfg_keys = config['keys']
    if mc is None:
        if args['mc'] is not None:
            mc_path = args['mc']
        else:
            mc_path = str(config['IC_MC']['path'])
        mc = np.load(str(mc_path))
        mc = mc_cut(mc, config)
        if inds is not None:
            print('Cut on given indices..')
            mc = mc[inds]
    settings, grid = model.setup_KDE(mc, cfg_keys)
    mc_conv = len(mc)
    print('Use {} mc events'.format(mc_conv))
    for key in settings.keys():
        settings[key]['name'] = key
    for key in bws.keys():
        settings[key]['bandwidth'] = bws[key]

    plaw = np.vectorize(powerlaw)

    # create binned pdf
    if args['weights'] == 'default':
        print('Use pre-calculated input weights')
        weights = mc['cur_weight']
    elif args['weights'] == 'pl':
        weights = mc[cfg_keys['ow']] * plaw(mc[cfg_keys['trueE']],
                                                  phi0=args['phi0'],
                                                  gamma=args['gamma'])
    elif args['weights'] == 'conv':
        weights = mc[cfg_keys['conv']]
    elif args['weights'] == 'conv+pl':
        #diff_weight = mc['orig_OW'] * plaw(mc['trueE'], phi0=args['phi0'],
        #                                   gamma=args['gamma'])
        weights = mc[cfg_keys['conv']] + mc[cfg_keys['astro']]
        print('Rates [1/yr]:')
        print(np.sum(mc[cfg_keys['conv']]) * np.pi * 1e7)
        print(np.sum(mc[cfg_keys['astro']]) * np.pi * 1e7)
    else:
        print('{} is not a valid weights argument'.format(args['weights']))
        sys.exit(0)

    inp_arr = [settings[key] for key in settings.keys()]
    if args['adaptive']:
        m_input = meerkat_input(inp_arr, weights, mc_conv=mc_conv)
        m_kde4d_fb = meerkat_kde(m_input)
        adtv_input = meerkat_input(inp_arr, weights, pdf_seed=m_kde4d_fb.kde,
                                   adaptive=True, mc_conv=mc_conv)
        m_kde4d = meerkat_kde(adtv_input)
    else:
        m_input = meerkat_input(inp_arr, weights, mc_conv=mc_conv)
        m_kde4d = meerkat_kde(m_input)

    nbins = args['eval_bins']
    eval_grid = OrderedDict()
    if grid is None:
        grid = {}

    for key in settings.keys():
        if key in grid.keys():
            if isinstance(grid[key], list):
                eval_grid[key] = np.linspace(grid[key][0],
                                             grid[key][1],
                                             nbins)
            elif isinstance(grid[key], numpy.ndarray):
                eval_grid[key] = grid[key]
        else:
            eval_grid[key] = np.linspace(settings[key]['range'][0],
                                         settings[key]['range'][1],
                                         nbins)
    print(eval_grid.keys())
    out_bins = [eval_grid[key] for key in settings.keys()]
    coords = np.array(list(itertools.product(*out_bins)))
    bws = [settings[key]['bandwidth'] for key in settings.keys()]

    print('Evaluate KDEs:')
    pdf_vals = np.asarray([m_kde4d.eval_point(coord) for coord in coords])
    shpe = np.ones(len(settings.keys()), dtype=int) * nbins
    pdf_vals = pdf_vals.reshape(*shpe)

    add_str = ''
    if args['weights'] != 'pl':
        add_str = '_' + args['weights']
    else:
        add_str = '_' + 'g_{}'.format(args['gamma'])
    if args['save_str'] != '':
        add_str = add_str + '_' + args['save_str']

    odict = dict({'vars': eval_grid.keys(),
                  'bins': out_bins,
                  'coords': coords,
                  'pdf_vals': pdf_vals,
                  'bw': bws})

    if not args['no_save']:
        with open(os.path.join(args['outfolder'], mname + add_str + '.pkl'), 'wb') as fp:
            pickle.dump(odict, fp)
    t1 = time.time()
    print('Finished after {} minutes'.format((t1 - t0) / 60))
    return odict


if __name__ == "__main__":
    create_KDE(parseArguments())
