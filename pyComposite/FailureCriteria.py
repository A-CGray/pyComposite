"""
==============================================================================
pyComposite Failure Criteria
==============================================================================
@File    :   FailureCriteria.py
@Date    :   2020/11/20
@Author  :   Alasdair Christison Gray
@Description : A collection of functions for computing various lamina failure criteria
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================


def TsaiHill(Sigma1, Sigma2, Sigma12, S1, S2, S12):
    return (Sigma1 / S1) ** 2 - (Sigma1 * Sigma2 / S1 ** 2) + (Sigma2 / S2) ** 2 + (Sigma12 / S12) ** 2