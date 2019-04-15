import numpy as np
import os
import importlib
import sys
import configparser


def GreatCircleDistance(ra_1, dec_1, ra_2, dec_2):
    '''Compute the great circle distance between two events'''
    '''All coordinates must be given in radians'''
    delta_dec = np.abs(dec_1 - dec_2)
    delta_ra = np.abs(ra_1 - ra_2)
    x = (np.sin(delta_dec / 2.))**2. + np.cos(dec_1) *\
        np.cos(dec_2) * (np.sin(delta_ra / 2.))**2.
    return 2. * np.arcsin(np.sqrt(x))


def powerlaw(Et, phi0=1.01*1e-18, gamma = 2.19):
    return phi0 * (Et / 1.e5) ** (-gamma)


def get_mids(bins):
    return (bins[1:] + bins[:-1]) / 2


def load_model(model_path):
    model_path = os.path.abspath(model_path)
    sys.path.append(os.path.dirname(model_path))
    mname = os.path.splitext(os.path.basename(model_path))[0]
    return importlib.import_module(mname), mname


def read_config():
    config = configparser.ConfigParser()
    if os.path.exists('./setting.cfg'):
        config.read('./setting.cfg')
    else:
        print('Config file missing')
    return config
