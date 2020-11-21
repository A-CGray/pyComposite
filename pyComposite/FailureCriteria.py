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


def TsaiHill(Sigma1, Sigma2, Sigma12, S1, S2, S12, returnSafetyFactor=False):
    """Compute the Tsai-Hill failure criterion for a plane stress state

    [extended_summary]

    Parameters
    ----------
    Sigma1 : float
        1-Direction axial stress
    Sigma2 : float
        2-Direction axial stress
    Sigma12 : float
        In-plane shear stress
    S1 : float
        1-Direction axial strength
    S2 : float
        2-Direction axial strength
    S12 : float
        In-plane shear strength
    returnSafetyFactor : bool, optional
        if True, the value returned is the factor of safety rather than the failure criterion, by default False

    Returns
    -------
    [type]
        [description]
    """
    FC = (Sigma1 / S1) ** 2.0 - (Sigma1 * Sigma2 / S1 ** 2.0) + (Sigma2 / S2) ** 2.0 + (Sigma12 / S12) ** 2.0
    if returnSafetyFactor:
        return np.sqrt(1.0 / FC)
    else:
        return FC
