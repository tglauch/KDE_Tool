import numpy as np

def diffuse_cuts(mc,config):
    keys = config['keys']
    logEt = np.log10(mc[keys['trueE']])
    return mc[(mc[keys['trueDec']] > np.radians(-5)) & (logEt < 8.0) &  (mc[keys['sigmaOK']] == 0)]
