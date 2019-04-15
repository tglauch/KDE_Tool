import numpy as np

def diffuse_cuts(mc):
    return mc[(mc['trueDec'] > np.radians(-5)) & (mc['logEt'] < 8.0) &  (mc['sigmaOK'] == 0)]
