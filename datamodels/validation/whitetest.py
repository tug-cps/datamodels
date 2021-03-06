from statsmodels.stats import diagnostic as diag
import numpy as np


def white_test(x_test, residual):
    """
    White's Lagrange Multiplier test for heteroscedascity
    :param x_test: Test input data
    :param residual: Residual values (x_true - x_predicted)
    :return: {'pvalue_lm', 'pvalue_f} - White's test statistics
    """
    [_,pvalue_lm,_,pvalue_f] = diag.het_white(residual, np.reshape(x_test,(x_test.shape[0], -1)))
    return {'pvalue_lm':pvalue_lm,'pvalue_f':pvalue_f}