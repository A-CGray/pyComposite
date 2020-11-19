"""
==============================================================================
pyLaminate: A python class for composite laminate plate calculations
==============================================================================
@File    :   pyLaminate.py
@Date    :   2020/11/19
@Author  :   Alasdair Christison Gray
@Description : This code implements the laminate class, which models a composite laminated plate consisting of a stack
of composite plies, each of which are modelled by pyComposite's lamina class.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import copy

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from . import pyCompTransform as transforms


class laminate(object):
    def __init__(self, plies, plyAngles, plyThicknesses, symmetric=True):
        """[summary]

        [extended_summary]

        Parameters
        ----------
        plies : pyLamina object or list of pyLamina objects
            The lamina for each layer of the laminate, if a single lamina is supplied then that lamina will be used for every layer
        plyAngles : list
            list of ply angles in degrees, must be the same length as the number of plies (or half the number if defining a symmetric laminate)
        plythicknesses : float or list
            the thickness of each ply, if a single value is supplied then this thickness is used for all plies.
        symmetric : bool, optional
            Whether the laminate is symmetric, if it is then the inputs are assumed to describe one half of the laminate, by default True
        """

        # First, create a list of the laminae, ply angles and ply thicknesses ignoring whether it is symmetric or not for now
        self.plyAngles = plyAngles

        if isinstance(plies, list):
            if len(plies) != len(plyAngles):
                raise ValueError(
                    "You must either supply a single pyLamina object or a list of pyLamina the same length as the supplied list of angles"
                )
            else:
                self.plies = plies
        else:
            self.plies = [copy.deepcopy(plies) for i in self.plyAngles]

        if isinstance(plyThicknesses, list):
            if len(plyThicknesses) != len(plyAngles):
                raise ValueError(
                    "You must either supply a single pyLamina object or a list of pyLamina the same length as the supplied list of angles"
                )
            else:
                self.plyThicknesses = plyThicknesses
        else:
            self.plyThicknesses = [copy.deepcopy(plyThicknesses) for i in self.plyAngles]

        # if the laminate symmetric then we want to create the plies, angles and thicknesses for the other half of the laminate
        if symmetric:
            for prop in [self.plies, self.plyThicknesses, self.plyAngles]:
                prop += prop[::-1]

        # COnvert thicknesses and angles to numpy arrays
        self.plyThicknesses = np.array(self.plyThicknesses)
        self.plyAngles = np.array(self.plyAngles)

        # Now compute the total thickness and z coordinates for each ply
        self.totalThickness = np.sum(self.plyThicknesses)
        self.zPlies = np.array([-self.totalThickness / 2.0])
        self.zPlies = np.concatenate((self.zPlies, np.cumsum(self.plyThicknesses) - self.totalThickness / 2.0))

    def computeCLTMat(self, matType="A", thermal=False):
        """Compute one of the laminate stiffness matrices

        This function can be used to create the A, B and D matrices

        Parameters
        ----------
        matType : str, optional
            [description], by default "A"
        thermal : bool, optional
            [description], by default False

        Returns
        -------
        [type]
            [description]
        """
        Mat = np.zeros(3) if thermal else np.zeros((3, 3))
        nDict = {"A": 1.0, "B": 2.0, "D": 3.0}
        n = nDict[matType.upper()]

        for i in range(len(self.plies)):
            ply = self.plies[i]
            angle = np.deg2rad(self.plyAngles[i])
            QBar = ply.getQBar(angle)
            if thermal:
                QBar = QBar @ transforms.TransformStrain(ply.CTEVec, -angle)
            Mat += 1 / n * QBar * (self.zPlies[i + 1] ** n - self.zPlies[i] ** n)
        return Mat

    def computeCLTStarMats(self):
        A = self.AMat
        B = self.BMat
        D = self.DMat
        AStar = np.linalg.inv(A)
        BStar = np.dot(-AStar, B)
        CStar = np.dot(B, AStar)
        DStar = D - np.dot(CStar, B)
        return AStar, BStar, CStar, DStar

    @property
    def AMat(self):
        return self.computeCLTMat(matType="A")

    @property
    def BMat(self):
        return self.computeCLTMat(matType="B")

    @property
    def DMat(self):
        return self.computeCLTMat(matType="D")

    @property
    def APrimeMat(self):
        return np.linalg.inv(self.AMat - self.BMat @ np.linalg.inv(self.DMat) @ self.BMat)

    @property
    def BPrimeMat(self):
        BDInv = self.BMat @ np.linalg.inv(self.DMat)
        return -np.linalg.inv(self.AMat - BDInv @ self.BMat) @ BDInv

    @property
    def DPrimeMat(self):
        B = self.BMat
        return np.linalg.inv(self.DMat - B @ np.linalg.inv(self.AMat) @ B)

    @property
    def EMatInv(self):
        return np.linalg.inv(self.EMat)

    @property
    def EMat(self):
        return np.block([[self.AMat, self.BMat], [self.BMat, self.DMat]])

    @property
    def NTVec(self):
        return self.computeCLTMat(matType="A", thermal=True)

    @property
    def MTVec(self):
        return self.computeCLTMat(matType="B", thermal=True)

    @property
    def NMTVec(self):
        return np.concatenate((self.NTVec, self.MTVec))

    @property
    def CTEVec(self):
        return np.dot(self.EMatInv, self.NMTVec)


if __name__ == "__main__":
    from pyLamina import lamina

    LaminaProps = {
        "E1": 138.0,
        "E2": 9.0,
        "v12": 0.3,
        "G12": 6.9,
        "CTE1": -0.3e-6,
        "CTE2": 28.1e-6,
    }

    ply = lamina(LaminaProps)
    print(f"QMat = \n{ply.QMat}\n")
    print(f"SMat = \n{ply.SMat}\n")

    angles = [45.0, -45.0]
    plyThick = 0.25
    Lam = laminate(ply, angles, plyThick, symmetric=False)

    print(f"Ply thicknesses = \n{Lam.plyThicknesses}\n")
    print(f"Ply angles = \n{Lam.plyAngles}\n")
    print(f"A Matrix = \n{Lam.AMat}\n")
    print(f"B Matrix = \n{Lam.BMat}\n")
    print(f"D Matrix = \n{Lam.DMat}\n")
    print(f"NT Vector = \n{Lam.NTVec}\n")
    print(f"MT Vector = \n{Lam.MTVec}\n")