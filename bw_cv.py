import numpy as np
from sklearn.model_selection import KFold
import argparse
from generate_kde import create_KDE
from functions import powerlaw, load_model, read_config
from scipy.interpolate import RegularGridInterpolator
import os
import sys
from numpy.lib.recfunctions import append_fields
from mc_cuts import diffuse_cuts as mc_cut

def parseArguments():
    """Parse the command line arguments
    Returns:
    args : Dictionary containing the command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_bins", type=int, default=100)
    parser.add_argument(
        "--gamma", type=float, default=2.19)
    parser.add_argument(
        "--phi0",
        help="in units of 1e-18 1/GeV/cm^2/sr/s",
        type=float, default=1.01)
    parser.add_argument(
        "--model", type=str, required=True)
    parser.add_argument(
        "--weights", type=str, default='pl')
    parser.add_argument(
        "--rs", type=int, default=None)
    parser.add_argument(
        "--save_str", type=str, default='')
    parser.add_argument(
        "--outfolder", type=str, default='./out')
    parser.add_argument(
        "--bw_key", nargs='+', type=str, required=True)
    parser.add_argument(
        "--bw", nargs='+', type=float, required=True)
    parser.add_argument(
        "--kfold", type=int, default=10)
    args = parser.parse_args()
    return args.__dict__


plaw = np.vectorize(powerlaw)


def do_validation(res_dict, settings, weights):
    bins = [res_dict['bins'][i] for i in range(len(res_dict['bins']))]
    pdf = RegularGridInterpolator(bins, res_dict['pdf_vals'],
                                  method='linear',
                                  bounds_error=False, fill_value=0)
    weights = weights / np.sum(weights)
    mc_arr = [settings[key]['values'] for key in settings.keys()]
    likelihood = pdf(list(zip(*mc_arr)))
    inds = likelihood > 0.
    return np.sum(np.log(likelihood[inds]) * weights[inds]), len(likelihood) - len(inds)


def cross_validate(args):
    assert len(args['bw_key']) == len(args['bw'])
    if not os.path.exists(args['outfolder']):
        os.makedirs(args['outfolder'])
    args['phi0'] *= 1e-18  # correct units
    kf = KFold(n_splits=args['kfold'], random_state=args['rs'], shuffle=True)
    config = read_config()
    print('Load MC: {}'.format(config['IC_MC']['path']))
    mc = np.load(str(config['IC_MC']['path']))[:]
    mc = mc_cut(mc)
    if args['weights'] == 'pl':
        weights = mc['orig_OW'] * plaw(mc['trueE'], phi0=args['phi0'],
                                       gamma=args['gamma'])
    elif args['weights'] == 'conv':
        weights = mc['conv']
    elif args['weights'] == 'conv+pl':
        diff_weight = mc['orig_OW'] * plaw(mc['trueE'], phi0=args['phi0'],
                                           gamma=args['gamma'])
        weights = mc['conv'] + diff_weight
        print('Rates [1/yr]:')
        print(np.sum(mc['conv']) * np.pi * 1e7)
        print(np.sum(diff_weight) * np.pi * 1e7)
    else:
        print('{} is not a valid weights argument'.format(args['weights']))
        sys.exit(0)
    mc = append_fields(mc, 'cur_weight', weights)
    args['weights'] = 'default'
    model, mname = load_model(args['model'])
    bw_dict = dict()
    for i, key in enumerate(args['bw_key']):
        bw_dict[key] = args['bw'][i]
    lh_arr, zero_arr = [], []
    for train_index, val_index in kf.split(mc):
        args['no_save'] = True
        res_dict = create_KDE(args, mc=mc[train_index], bws=bw_dict)
        mc_val = mc[val_index]
        val_settings, grid = model.setup_KDE(mc_val)
        lh, zeros = do_validation(res_dict, val_settings, mc_val['cur_weight'])
        print('Number of zeros {}'.format(zeros))
        print('Likelihood Value {}'.format(lh))
        zero_arr.append(zeros)
        lh_arr.append(lh)
    fname = ''
    for i in range(len(args['bw'])):
        fname += '{}_{}_'.format(args['bw_key'][i], args['bw'][i])
    fname = fname[:-1] + '.npy'
    odict = {'zeros': zero_arr, 'lh': lh_arr}
    np.save(os.path.join(args['outfolder'], fname), odict)


if __name__ == "__main__":
    cross_validate(parseArguments())
