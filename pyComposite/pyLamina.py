"""pyLamina: a class representing a single orthotropic lamina

[extended_summary]
"""

import numpy as np
from . import pyCompTransform as transforms
import warnings


class lamina(object):
    def __init__(self, matPropDict):
        self.MatProps = matPropDict

    @property
    def QMat(self):
        """Compute the lamina's stiffness matrix

        This matrix can be used to compute stresses from strains, all in the material frame:
        [ s_1  ]   [ Q_11, Q_12, 0    ] [    e_1   ]
        [ s_2  ] = [ Q_12, Q_22, 0    ] [    e_2   ]
        [Tau_12]   [ 0,    0,    Q_66 ] [ gamma_12 ]

        Returns
        -------
        3 x 3 array
            Stiffness matrix
        """

        QMat = np.zeros((3, 3))
        props = self.MatProps
        v21 = props["v12"] * props["E2"] / props["E1"]
        denom = 1 - props["v12"] * v21

        QMat[0, 0] = props["E1"] / denom
        QMat[1, 1] = props["E2"] / denom
        QMat[[1, 0], [0, 1]] = props["v12"] * props["E2"] / denom
        QMat[-1, -1] = props["G12"]

        return QMat

    @property
    def SMat(self):
        """Compute the lamina's compliance matrix

        This matrix can be used to compute strains from stresses, all in the material frame:
        [    e_1   ]   [ S_11, S_12, 0    ] [ s_1  ]
        [    e_2   ] = [ S_12, S_22, 0    ] [ s_2  ]
        [ gamma_12 ]   [ 0,    0,    S_66 ] [Tau_12]

        Returns
        -------
        3 x 3 array
            Stiffness matrix
        """

        SMat = np.zeros((3, 3))
        props = self.MatProps

        SMat[0, 0] = 1.0 / props["E1"]
        SMat[1, 1] = 1.0 / props["E2"]
        SMat[[1, 0], [0, 1]] = -props["v12"] / props["E1"]
        SMat[-1, -1] = 1.0 / props["G12"]

        return SMat

    @property
    def CTEVec(self):
        """Return the vector of thermal expansions coefficients, assuming the thermal expansions coefficients for
        the lamina are defined

        [extended_summary]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        KeyError
            [description]
        """
        try:
            return np.array([self.MatProps["CTE1"], self.MatProps["CTE2"], 0.0])
        except KeyError:
            warnings.warn(
                "Lamina thermal expansion properties requested but no thermal expension properties defined, returning zero vector"
            )
            return np.zeros(3)

    def getQBar(self, theta):
        if theta % np.pi == 0:
            return self.QMat
        else:
            return transforms.TransformStiffnessMat(self.QMat, theta)

    def getSBar(self, theta):
        if theta % np.pi == 0:
            return self.SMat
        else:
            return transforms.TransformComplianceMat(self.SMat, theta)


if __name__ == "__main__":
    LaminaProps = {
        "E1": 6.894 * 22.99,
        "E2": 6.894 * 1.41,
        "v12": 0.316,
        "G12": 6.894 * 0.68,
        "CTE1": -0.3e-6,
        "CTE2": 28.1e-6,
    }
    Lam = lamina(LaminaProps)
    print(f"QMat = {Lam.QMat}")
    print(f"SMat = {Lam.SMat}")