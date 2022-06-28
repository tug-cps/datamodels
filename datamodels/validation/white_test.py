from statsmodels.stats import diagnostic as diag


def white_test(x_test, residual):
    """
    White's Lagrange Multiplier test for heteroscedascity
    :param x_test: Test input data
    :param residual: Residual values (x_true - x_predicted)
    :return: {'pvalue_lm', 'pvalue_f} - White's test statistics
    """
    # White - Test
    try:
        [_,pvalue_lm,_,pvalue_f] = diag.het_white(residual, x_test.reshape(x_test.shape[0], -1))
        return {'pvalue_lm':pvalue_lm,'pvalue_f':pvalue_f}
    except AssertionError:
        print('White Test: Assertion error during calculation. Results are invalid.')
    return {}