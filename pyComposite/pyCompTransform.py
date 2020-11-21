"""Various functions for performing coordinate transformations for basic composites analysis

[extended_summary]
"""
import numpy as np


def TMat(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c ** 2, s ** 2, 2 * c * s], [s ** 2, c ** 2, -2 * c * s], [-s * c, s * c, c ** 2 - s ** 2]])


def TMatT(theta):
    return np.transpose(TMat(theta))


def TMatInv(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c ** 2, s ** 2, -2 * c * s], [s ** 2, c ** 2, 2 * c * s], [s * c, -s * c, c ** 2 - s ** 2]])


def TMatInvT(theta):
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c ** 2, s ** 2, s * c], [s ** 2, c ** 2, -s * c], [-2 * s * c, 2 * s * c, c ** 2 - s ** 2]])


def TransformStrain(strain, theta):
    """Rotate a state of 2D strain through and angle theta

    This method assumes the input strain uses engineering shear strain. It thus converts to tensor shear strain before
    transforming and then converts back to engineering shear before returning the strain state

    Parameters
    ----------
    strain : array
        Strain state in original coordinate frame [e_1, e_2, gamma_12]
    theta : float
        Angle by which to rotate the the strain, in radians. Positive theta represents a CCW rotation around x3

    Returns
    -------
    transformedStrain : array
        Strain in the rotated coordinate frame, [e1', e2', gamma_12']
    """

    return np.dot(TMatInvT(theta), strain)


def TransformStress(stress, theta):
    """Rotate a plane stress state through and angle theta

    This method assumes the input strain uses engineering shear strain. It thus converts to tensor shear strain before transforming and then converts back to engineering shear before returning the strain state

    Parameters
    ----------
    stress : array
        stress state in original coordinate frame [s_1, s_2, tau_12]
    theta : float
        Angle by which to rotate the the stress, in radians. Positive theta represents a CCW rotation around x3

    Returns
    -------
    array
        stress in the rotated coordinate frame, [s1', s2', tau_12']
    """
    return np.dot(TMat(theta), stress)


def TransformStiffnessMat(QMat, theta):
    """Compute the stiffness matrix of a material in a rotated coordinate frame

    This method assumes the stiffness matrix is in the form which computes stress based on engineering shear strain rather than tensor shear strain.

    Parameters
    ----------
    QMat : 3x3 array
        Material stiffness matrix in original coordinate frame
    theta : float
        Angle between coordinate systems, in radians. Positive theta represents a CCW rotation around x3 from reference to transformed coordinates

    Returns
    -------
    QBarMat : 3x3 array
        Material stiffness matrix in rotated coordinate system, QBar
    """
    return TMatInv(theta) @ QMat @ TMatInvT(theta)


def TransformComplianceMat(SMat, theta):
    """Compute the compliance matrix of a material in a rotated coordinate frame

    This method assumes the compliance matrix is in the form which directly computes engineering shear strain rather than tensor shear strain.

    Parameters
    ----------
    SMat : 3x3 array
        Material compliance matrix in original coordinate frame
    theta : float
        Angle between coordinate systems, in radians. Positive theta represents a CCW rotation around x3 from reference to transformed coordinates

    Returns
    -------
    SBarMat : 3x3 array
        Material compliance matrix in rotated coordinate system, SBar
    """
    return TMatT(theta) @ SMat @ TMat(theta)