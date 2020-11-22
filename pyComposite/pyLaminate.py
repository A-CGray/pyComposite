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

# ==============================================================================
# Wish List
# ==============================================================================
# TODO: Add visualisation methods for:
# Polar stiffness plots
# Stress strain curves
# Ply orientations


class laminate(object):
    def __init__(self, plies, plyAngles, plyThicknesses, symmetric=True):
        """Create a model of a composite laminate, consisting of multiple composite lamina

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
        self.plyAngles = copy.deepcopy(plyAngles)

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

        # Convert thicknesses and angles to numpy arrays
        self.plyThicknesses = np.array(self.plyThicknesses)
        self.plyAngles = np.array(self.plyAngles)

        # Create a list of failure flags for each ply, these can be used to exclude the contribution of a ply to the
        # laminate stiffness
        self.plyFailed = [False for i in self.plyAngles]

        @property
        def totalThickness(self):
            """Return the total thickness of the laminate"""
            return np.sum(self.plyThicknesses)

        @property
        def zPlies(self):
            """Compute the z coordinates of the start and end of each ply

            z = 0 is the top surface of the laminate

            Returns
            -------
            [type]
                [description]
            """
            zPlies = np.array([-self.totalThickness / 2.0])
            return np.concatenate((zPlies, np.cumsum(self.plyThicknesses) - self.totalThickness / 2.0))

    def computeCLTMat(self, matType="A", thermal=False):
        """Compute one of the laminate stiffness matrices

        This function can be used to create the A, B and D matrices of the laminate, or by enabling the 'thermal'
        option, the thermal load vectors NT and MT.

        Parameters
        ----------
        matType : str, optional
            Which laminate matrix to compute, with the 'thermal' option enabled, 'A' will compute the thermal force
            vector NT and 'B' will compute the thermal moment vector MT, by default "A"
        thermal : bool, optional
            set True to compute thermal force or moment vector as opposed to A, B or D matrix, by default False

        Returns
        -------
        array
            The matrix or vector requested
        """

        Mat = np.zeros(3) if thermal else np.zeros((3, 3))

        n = 1.0 if matType.upper() == "A" else 2.0 if matType.upper() == "B" else 3.0

        for i in range(len(self.plies)):
            if not self.plyFailed[i]:
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

    @property
    def flexModulii(self):
        "Return a vector containing the 3 flexural modulii of the panel, E_fx, E_fy and E_fxy"
        return 12.0 / (self.totalThickness ** 3 * np.diagonal(self.DPrimeMat))

    @property
    def inPlaneModulii(self):
        "Return a vector containing the 3 in-Plane modulii of the panel, E_x, E_y and G_xy"
        return 1.0 / (self.totalThickness * np.diagonal(self.APrimeMat))

    def computeLaminateStrain(self, N=None, M=None, deltaT=0.0):
        """Compute mid-plane strain and curvature in laminate under given mechanical and thermal load

        Parameters
        ----------
        N : array, optional
            applied force vector N, in units of force/distance, by default None, which is taken to mean no load is applied
        M : array, optional
            Applied moment vector M, in units of force, by default None, which is taken to mean no moment is applied
        deltaT : float, optional
            Temperature change, used to compute thermal expansion loads, by default 0.

        Returns
        -------
        e : array
            Mid-plane strains, [e_x, e_y, gamma_xy]
        k : array
            Mid-plane curvatures, [kappa_x, kappa_y, kappa_xy]
        """
        NMVec = np.zeros(6)
        if N is not None:
            NMVec[:3] = N
        if M is not None:
            NMVec[3:] = M
        if deltaT != 0.0:
            NMVec += deltaT * self.NMTVec

        ekVec = self.EMat @ NMVec
        e = np.copy(ekVec[:3])
        k = np.copy(ekVec[3:])
        return e, k

    def computeStrainAtZ(self, z, e=None, k=None):
        """Calculate the strain at a given height within the laminate based on the mid-plane strains and curvatures

        Parameters
        ----------
        z : float or array
            Through thickness coordinate at which to compute strain, where z=0 is the laminate mid-plane and z=-t/2 is the top surface
        e : array, optional
            Mid-plane laminate strains, by default None, which case mid-plane strains will be taken to be zero
        k : array, optional
            Mid-plane laminate curvatures, by default None, which case mid-plane curvatures will be taken to be zero

        Returns
        -------
        strains : arrays
            2D array where each row is 2D strain state [e_x, e_y, gamma_xy] at the corresponding z coordinate
        """

        if e is None:
            e = np.zeros(3)
        if k is None:
            k = np.zeros(3)
        return e + np.outer(z, k)

    def computeUltimateLoadFactors(self, N=None, M=None, printResults=False):
        """Perform a sequential failure analysis on the laminate for a given loading

        Given a specific force and moment vector, this function will compute the complete failure history of the laminate.
        This is done by computing the scaling of the given load under which failure occurs in any of the plies, then removing
        this ply's contribution to the stiffness of the laminate and repeating the process until all plies have failed.

        Parameters
        ----------
        N : array, optional
            applied force vector N, in units of force/distance, by default None, which is taken to mean no load is applied
        M : array, optional
            Applied moment vector M, in units of force, by default None, which is taken to mean no moment is applied

        Returns
        -------
        loadFactors : list
            The load factors at which each failure event occurs
        plyFailures : list of lists
            A list containing lists of the plies which fail at each failure event
        laminateStrains : list of arrays
            A list containing the laminate mid-plane strains and curvatures at each failure point
        """
        loadFactors = []
        plyFailures = []
        laminateStrains = []

        startFailedPlies = copy.deepcopy(self.plyFailed)

        if printResults:
            print("Laminate failure analysis:")
            print("--------------------------")
            print("Starting with these plies already failed:")
            print(startFailedPlies)

        n = 0
        while not all(self.plyFailed):

            n += 1

            # Compute laminate strains
            e, k = self.computeLaminateStrain(N, M)

            # compute strains at all the ply z coordinates
            strains = self.computeStrainAtZ(self.zPlies, e, k)

            # Compute safety factor at top and bottom edge of all plies and take minimum
            plySafetyFactors = []
            for i in range(len(self.plies)):
                if not self.plyFailed[i]:
                    ply = self.plies[i]
                    theta = np.deg2rad(self.plyAngles[i])
                    plyFailure = 1e6
                    for j in [i.i + 1]:
                        laminaFrameStrain = transforms.TransformStrain(strains[j], theta)
                        laminaFrameStress = ply.QMat @ laminaFrameStrain
                        plyFailure = min(plyFailure, ply.TsaiHillCriteria(laminaFrameStress, returnSafetyFactor=True))
                    plySafetyFactors.append(plyFailure)
                else:
                    plySafetyFactors.append(np.inf)

            # Find lamina with lowest load factor (may be multiple) and their fail flag to true
            minSafetyFactor = np.min(plySafetyFactors)
            failingPlies = np.where(plySafetyFactors == minSafetyFactor)
            self.plyFailed[failingPlies] = True

            # Save the load factor at which these ply failures will occur, which plies fail, and the laminate strain
            loadFactors.append(minSafetyFactor)
            plyFailures.append(failingPlies)
            laminateStrains.append(np.concatenate((e, k)) * minSafetyFactor)

            if printResults:
                print(f"\nFailure Stage {n}:")
                print(f"Load Factor: {loadFactors[-1]}")
                print(f"Failing Plies {plyFailures[-1]}")
                print(f"Mid-Plane Strains at failure: {laminateStrains[-1][:3]}")
                print(f"Mid-Plane Curvatures at failure: {laminateStrains[-1][3:]}")

        # Before exiting, reset the ply failure flags to whatever they were before this analysis
        self.plyFailed = startFailedPlies

        return loadFactors, plyFailures, laminateStrains
