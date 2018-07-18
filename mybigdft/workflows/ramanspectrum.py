r"""
File containing the class allowing to compute the Raman Spectrum of a
molecular system with BigDFT.
"""

from __future__ import print_function, absolute_import
from copy import deepcopy
import os
import numpy as np
from .mybigdft import BigDFTCalc, COORDS, SIGNS
from .poltensor import PolTensorCalc
from .iofiles import Logfile


# Mass of the different types of atoms in atomic mass units
# TODO: Add more types of atoms
#       (found in $SRC_DIR/bigdft/src/orbitals/eleconf-inc.f90)
MASS_ATOMS = {"H": 1.00794, "He": 4.002602, "Li": 6.941, "Be": 9.012182,
              "B": 10.811, "C": 12.011, "N": 14.00674, "O": 15.9994,
              "F": 18.9984032, "Ne": 20.1797}
# Conversion from atomic to electronic mass unit
AMU_TO_EMU = 1.660538782e-27 / 9.10938215e-31
# Conversion from electronic to atomic mass unit
EMU_TO_AMU = 1. / AMU_TO_EMU
# Conversion factor from bohr to angstroem
B_TO_ANG = 0.529177249
# Conversion factor from angstroem to bohr
ANG_TO_B = 1. / B_TO_ANG
# Conversion factor from Hartree to cm^-1
HA_TO_CMM1 = 219474.6313705


class RamanSpectrumCalc(BigDFTCalc):
    r"""
    This class allows one to initialize and run a Raman spectrum
    calculation with BigDFT. The main quantities computed are
    the phonon energies, the intensity of the peaks, and the
    depolarization ratio.

    See http://dx.doi.org/10.1063/1.470783 and
    https://link.aps.org/doi/10.1103/PhysRevB.54.7830 for more
    details on how to obtain these quantities from a set of
    calculations.
    """

    def __init__(self, input_yaml, posinp, alpha_x=1./64.,
            calc_intensities=True, ef_amplitudes=[1.E-4]*3, prefix=None,  # noqa
            run_folder=None, ref_calc=None):  # noqa
        r"""
        Method initializing a Raman spectrum calculation. It inherits
        from a BigDFTCalc object.

        From a single equilibrium geometry posinp and a standard input
        file, it launches calculations where each atom is moved around
        its initial position in the three space coordinates to get the
        vibrational energies. This amounts to 6 n_at BigDFT
        calculations (where n_at is the number of atoms of the system).

        The distance of the displacement in a direction is controlled
        by the grid space hgrid in that direction multiplied by
        alpha_x. These 6 n_at calculations allow only for the phonon
        energies calculations as post-processing.

        If interested in getting the intensity and depolarization
        ratio of each normal mode, set calc_intensities to True. This
        requires 36 n_at extra calculations in order to get the
        polarizability tensor for each geometry around the equilibrium
        positions.

        The reference calculation must correspond to the equilibrium
        calculation geometry.

        :param input_yaml: Reference input file for all the
                           calculations.
        :type input_yaml: inputfile.Input
        :param posinp: Reference geometry for all the calculations.
        :type posinp: posinp.Posinp
        """
        # Check that the reference calculation corresponds to the
        # equilibrium geometry
        if ref_calc is not None:
            if ref_calc.posinp != posinp:
                raise ValueError("The initial geometry of the reference \
                        calculation does not correspond to the \
                        equilibrium geometry.")

        # Initialize the instance via the BigDFTCalc class
        super(RamanSpectrumCalc, self).__init__(input_yaml, posinp,
            prefix=prefix, run_folder=run_folder, ref_calc=ref_calc)  # noqa

        # Initialize the norm of the displacements along the three
        # space coordinates. It corresponds to grid step along the
        # three direction multiplied by alpha_x. Unit in bohr.
        # If no hgrids are present in the input file, use a default
        # value and update the input file.
        if 'dft' not in input_yaml:
            input_yaml['dft'] = {}
        if 'hgrids' not in input_yaml['dft']:
            hgrids = np.array([0.45]*3)
            input_yaml['dft']['hgrids'] = hgrids
        else:
            hgrids = input_yaml['dft']['hgrids']
            if not isinstance(hgrids, list):
                hgrids = [hgrids] * 3
            hgrids = np.array(hgrids)
        self.displacements = hgrids * alpha_x

        # Define if intensities must be calculated.
        if (posinp.BC == 'periodic' and calc_intensities) \
            or (posinp.BC == 'surface' and calc_intensities
                and ef_amplitudes[0] is not None
                and ef_amplitudes[2] is not None):
            raise UserWarning(
                "calc_intensities set to False. The posinp must use free "
                "boundary conditions for the intensities to be computed.")
        else:
            self.calc_intensities = calc_intensities

        # Initialize the name of the folders for each atom
        if posinp.n_at <= 999:
            self.atom_dirs = ["atom_{:03d}".format(i_at)
                              for i_at in range(posinp.n_at)]
        else:
            raise ValueError("Too many atoms in the system, modify the \
                previous lines of the code for larger systems to be \
                accessible.")

        if calc_intensities:
            # Initialize the electric field amplitudes
            self.ef_amplitudes = ef_amplitudes

    def run(self, nmpi=1, nomp=1, force_run=False, dry_run=False):
        r"""
        Method running all the necessary calculations and
        post-processing the logfiles to set all the quantities of
        interest (energies, intensities, depolarization ratios, ...).

        :param nmpi: Number of MPI tasks
        :type nmpi: int
        :param nomp: Number of OpenMP tasks.
        :type nomp: int
        :param force_run: States if the calculation has to be run,
                          even though a logfile already exists.
                          (Optional, default value set to False)
        :type force_run: boolean
        :param dry_run: If True, all folders and input files are written
            on disk, but are not run.
        :type dry_run: bool
        """
        # Initialize a new input file so that BigDFT outputs the
        # wavefunctions.
        new_input = deepcopy(self.input_yaml)
        # if self.calc_intensities:
        #     new_input['output'] = {'orbitals': True}
        # else:
        #     new_input['output'] = {'orbitals': False}
        # Run the reference calculation
        if self.ref_calc is None:
            run_folder = os.path.join(self.run_folder, "ref")
            self.ref_calc = BigDFTCalc(new_input, self.posinp,
                              prefix=self.prefix, run_folder=run_folder)  # noqa
        self.ref_calc.run(nmpi=nmpi, nomp=nomp, force_run=force_run,
                          dry_run=dry_run)
        # Initialize the dictionary of paths leading to the relevant
        # logfiles and the dictionary of polarizability tensors
        self.logfiles_paths = {}
        if self.calc_intensities:
            self.pol_tensors = {}
        # Loop over the atoms
        for i_at, atom_dir in enumerate(self.atom_dirs):
            # The logfiles and polarizability tensors concerning the
            # displacements of the current atom will be stored in
            # dictionaries
            self.logfiles_paths[atom_dir] = {}
            if self.calc_intensities:
                self.pol_tensors[atom_dir] = {}
            # Loop over the space coordinates
            for i, coord in enumerate(COORDS):
                # The two logfiles and polarizability tensors concerning
                # the atomic displacements along the same space
                # coordinate of the same atom will be stored in lists
                self.logfiles_paths[atom_dir][coord] = []
                if self.calc_intensities:
                    self.pol_tensors[atom_dir][coord] = []
                # Loop over the sign of the displacement
                for sign in SIGNS:
                    # Update the posinp (convert the translation
                    # vector in the correct units before that)
                    vector = [0.0]*3
                    vector[i] = SIGNS[sign] * self.displacements[i]
                    units = self.posinp.units
                    if 'angstroem' in units:
                        vector = [val*B_TO_ANG for val in vector]
                    elif all([elem not in units
                              for elem in ['bohr', 'atomic']]):
                        raise NotImplementedError(
                            "Conversion from bohr to {} not possible yet."
                            .format(units))
                    new_posinp = self.posinp.translate_atom(i_at, vector)
                    # Set the run_folder of the new BigDFT calculation
                    mv_dir = "{}{}".format(coord, sign)
                    run_folder = os.path.join(self.run_folder, atom_dir,
                                              mv_dir)
                    # Run the calculation for the new geometry
                    bdft = BigDFTCalc(new_input, new_posinp,
                            prefix=self.prefix, ref_calc=self.ref_calc,
                            run_folder=run_folder)  # noqa
                    bdft.run(nmpi=nmpi, nomp=nomp, force_run=force_run,
                             dry_run=dry_run)
                    # Update the dictionary of paths to the logfiles
                    self.logfiles_paths[atom_dir][coord].append(
                        bdft.logfile_path)
                    # If the intensities have to be computed, compute
                    # the polarizability tensor for the given geometry
                    if self.calc_intensities:
                        # Use the reference input file and the new
                        # posinp in the same run_folder. Use the
                        # previous bigdft calculation as a reference.
                        pt_calc = PolTensorCalc(self.input_yaml,
                            new_posinp, ef_amplitudes=self.ef_amplitudes,
                            prefix=self.prefix, run_folder=run_folder,
                            ref_calc=bdft)  # noqa
                        pt_calc.run(nmpi=nmpi, nomp=nomp, force_run=force_run,
                                    dry_run=dry_run)
                        # Update the dictionary of polarizability tensors
                        self.pol_tensors[atom_dir][coord].append(
                            pt_calc.pol_tensor)
        # Post-processing:
        if not dry_run:
            # - Set the Hessian
            self.hessian = self.build_hessian()
            # - Set the dynamical matrix
            self.dyn_mat = self.build_dyn_mat()
            # - Set the energies
            self.energies = {}
            self.energies['Ha'], self.normal_modes = \
                self.solve_dynamical_matrix()
            self.energies['cm^-1'] = self.energies['Ha'] * HA_TO_CMM1
            # - Extra post-processing if the intensities and
            #   depolarization were to be computed:
            if self.calc_intensities:
                # - Set the derivatives of the polarizability tensors
                #   along each displacement directions
                self.deriv_pol_tensors = self.find_deriv_pol_tensors()
                # - Set the mean polarizability derivatives (alpha), the
                #   anisotropies of the polarizability tensor derivative
                #   (beta_sq), the intensity and the depolarization ratio
                #   for each normal mode:
                self.alphas = []
                self.betas_sq = []
                self.intensities = []
                self.depol_ratios = []
                # - Loop over the normal modes
                for pt_flat in self.deriv_pol_tensors.dot(self.normal_modes).T:
                    # Reshape the derivative of the polarizability tensor
                    # along the current normal mode
                    pt = pt_flat.reshape(3, 3)
                    # Find the principal values of polarizability
                    alphas = np.linalg.eigvals(pt)
                    # Mean polarizability derivative
                    alpha = np.sum(alphas) / 3.
                    self.alphas.append(alpha)
                    # Anisotropy of the polarizability tensor derivative
                    beta_sq = ((alphas[0]-alphas[1])**2 +
                               (alphas[1]-alphas[2])**2 +
                               (alphas[2]-alphas[0])**2) / 2.
                    self.betas_sq.append(beta_sq)
                    # # Mean polarizability derivative
                    # alpha = 1./3. * pt.trace()
                    # self.alphas.append(alpha)
                    # # Anisotropy of the polarizability tensor derivative
                    # beta_sq = 1./2. * ((pt[0][0]-pt[1][1])**2 +
                    #                    (pt[0][0]-pt[2][2])**2 +
                    #                    (pt[1][1]-pt[2][2])**2 +
                    #                    6.*(pt[0][1]**2+pt[0][2]**2+pt[1][2]**2))
                    # self.betas_sq.append(beta_sq)
                    # From the two previous quantities, it is possible to
                    # compute the intensity (converted from atomic units
                    # to Ang^4.amu^-1) and the depolarization ratio
                    # of the normal mode.
#                    self.intensities.append(45*alpha**2 + 7*beta_sq)
                    conversion = B_TO_ANG**4 / EMU_TO_AMU
                    self.intensities.append(
                        (45*alpha**2 + 7*beta_sq) * conversion)
                    self.depol_ratios.append(
                        3*beta_sq / (45*alpha**2 + 4*beta_sq))

    def solve_dynamical_matrix(self):
        r"""
        Method solving the dynamical matrix to get the phonon energies
        (converted in Hartree) and the eigenvectors.

        :returns: Tuple made of the eigenvalues (as an array) and the
                  eigenvectors (as a matrix).
        :rtype: tuple
        """
        eigs, vecs = np.linalg.eig(self.dyn_mat)
        eigs = np.array([np.sqrt(-e) if e < 0 else np.sqrt(e) for e in eigs])
        return eigs, vecs

    def build_dyn_mat(self):
        r"""
        Method computing the dynamical matrix of the system. It is
        very similar to the Hessian matrix: its elements are only
        corrected by a weight w, which is the inverse of the sqrt of
        the product of the atomic masses of the atoms involved in the
        Hessian matrix element H[i][j]:

        w[i][j] = 1 / \sqrt(mass_i * mass_j)

        where mass_i is is the mass of the atom indexed by i (running
        from 1 to the number of atoms n_at).

        The masses are counted in electronic mass units (which is the
        atomic unit of mass, that is different from the atomic mass
        unit).

        :returns: Dynamical matrix
        :rtype: 2D square np.array of dimension 3*n_at
        """
        # Numpy does the ratio of arrays intellegently: by making
        # masses an array of the same size as the Hessian, there is
        # nothing but the ratio of both arrays to perform to get
        # the dynamical matrix.
        h = self.build_hessian()
        masses = self.build_masses()
        return h/masses

    def build_masses(self):
        r"""
        Method computing the masses array used to define the dynamical
        matrix. The masses are counted in electronic mass units (which
        is the atomic unit of mass, that is different from the atomic
        mass unit).

        :returns: Masses matrix
        :rtype: 2D square np.array of dimension 3*n_at
        """
        # Get the atoms of the system from the reference posinp
        atoms = self.posinp.atoms
        # Build the masses matrix (the loops over range(3) are here
        # to ensure that masses has the same dimension as the Hessian)
        masses = [[np.sqrt(MASS_ATOMS[atom1["Type"]] *
                           MASS_ATOMS[atom2["Type"]])
                   for atom2 in atoms for j in range(3)]
                  for atom1 in atoms for i in range(3)]
        # Return the masses as a numpy array, converted in electronic
        # mass units
        return np.array(masses)*AMU_TO_EMU

    def build_hessian(self):
        r"""
        Method computing the Hessian of the system. Its size is 3*n_at
        by 3*n_at, where n_at is the number of atoms of the system.

        In the case of a diatomic molecule, the Hessian matrix H is
        given by:

            [d F_{x}_1   d F_{y}_1   d F_{z}_1   d F_{x}_2   d F_{y}_2   d F_{z}_2]
            [---------   ---------   ---------   ---------   ---------   ---------]
            [   d x_1      d x_1       d x_1       d x_1       d x_1       d x_1  ]
            [                                                                     ]
            [d F_{x}_1   d F_{y}_1   d F_{z}_1   d F_{x}_2   d F_{y}_2   d F_{z}_2]
            [---------   ---------   ---------   ---------   ---------   ---------]
            [   d y_1      d y_1       d y_1       d y_1       d y_1       d y_1  ]
            [                                                                     ]
            [d F_{x}_1   d F_{y}_1   d F_{z}_1   d F_{x}_2   d F_{y}_2   d F_{z}_2]
            [---------   ---------   ---------   ---------   ---------   ---------]
            [   d z_1      d z_1       d z_1       d z_1       d z_1       d z_1  ]
        H = [                                                                     ]
            [d F_{x}_1   d F_{y}_1   d F_{z}_1   d F_{x}_2   d F_{y}_2   d F_{z}_2]
            [---------   ---------   ---------   ---------   ---------   ---------]
            [   d x_2      d x_2       d x_2       d x_2       d x_2       d x_2  ]
            [                                                                     ]
            [d F_{x}_1   d F_{y}_1   d F_{z}_1   d F_{x}_2   d F_{y}_2   d F_{z}_2]
            [---------   ---------   ---------   ---------   ---------   ---------]
            [   d y_2      d y_2       d y_2       d y_2       d y_2       d y_2  ]
            [                                                                     ]
            [d F_{x}_1   d F_{y}_1   d F_{z}_1   d F_{x}_2   d F_{y}_2   d F_{z}_2]
            [---------   ---------   ---------   ---------   ---------   ---------]
            [   d z_2      d z_2       d z_2       d z_2       d z_2       d z_2  ]

        where d actually denotes a partial dericative, F_{q}_{j} is
        the force along the coordinate q (x, y or z) and j is is the
        atom index, running from 1 to the number of atoms n_at
        (n_at = 2 here). d q_i denotes the displacement of the atom i
        along the coordinate q.

        :returns: Hessian matrix
        :rtype: 2D square np.array of dimension 3*n_at
        """  # noqa
        # Initialization of variables
        h = []  # Hessian matrix
        n_at = self.posinp.n_at  # Number of atoms
        # First loop over all atoms
        # for i_at, atom_dir in enumerate(self.atom_dirs):
        for atom_dir in self.atom_dirs:
            # Loop over the coordinates (x, y and z)
            for i, coord in enumerate(COORDS):
                # Read the two logfiles corresponding to the moves
                # along the same direction (+ and -)
                logs = [Logfile(self.logfiles_paths[atom_dir][coord][j])
                        for j in range(2)]
                # The Hessian is made of the delta of the forces
                # with respect to the delta of the move distances.
                # It is built line by line:
                new_line = []
                # 1- Find the delta displacement. It is twice the
                #    distance of the positive move along the direction
                #    of the displacement.
                delta_x = 2 * self.displacements[i]
                # 2- Find the delta forces for each atom and update
                #    the new line of the Hessian.
                for j_at in range(n_at):
                    forces = [np.array(log.log['Atomic Forces (Ha/Bohr)'][j_at].values()[0])  # noqa
                              for log in logs]
                    # forces = [np.array(log.forces[j_at].values()[0])
                    #           for log in logs]
                    delta_forces = forces[0] - forces[1]
                    new_line += list(delta_forces/delta_x)
                # The new line of the Hessian is now complete
                h.append(new_line)
        # Return the symmetrized Hessian matrix as a numpy array
        h = np.array(h)
        # Ensure the symmetry of the Hessian matrix
        h = (h + h.T) / 2.
        # Enforce the acoustic sum rule
        for i, row in enumerate(h):
            h[i][i] = -(np.sum(row) - h[i][i])
        return h  # (h + h.T) / 2.

    def find_deriv_pol_tensors(self):
        r"""
        Method computing the derivative of the polarizability tensor
        along all the atom displacements.

        All the elements of the derivative of the polarizability
        tensor along one displacement direction are represented by a
        line of the returned array. There are 3 * n_at such lines
        (because there are 3 displacements per atom). This
        representation allows for a simpler evaluation of these
        derivatives along the normal modes.

        Note that each element is also weighted by the inverse of the
        square root of the atom that is moved.

        :returns: Derivatives of the polarizability tensor.
        :rtype: 2D np.array of dimension (3*n_at) * 9
        """
        deriv_pol_tensors = []
        # Loop over the atoms
        for i_at, atom_dir in enumerate(self.atom_dirs):
            # Loop over the displacement directions
            for i, coord in enumerate(COORDS):
                # Get the delta of the polarizability tensors
                # corresponding to the two atomic displacements along
                # the current direction
                m = MASS_ATOMS[self.posinp.atoms[i_at]['Type']]
                pts = [self.pol_tensors[atom_dir][coord][j] for j in range(2)]
                delta_pol_tensor = (pts[0] - pts[1]) / np.sqrt(m*AMU_TO_EMU)
                # Norm of the atom displacement in that direction (in
                # bohr)
                delta_x = 2 * self.displacements[i]
                # Compute the derivative of each component of the
                # polarizability tensor for a given direction
                deriv = delta_pol_tensor / delta_x
                # The polarizability tensor is flatten to make it a
                # line of the derivative of the polarizability tensor
                # array.
                deriv_pol_tensors.append(deriv.flatten())
        # Convert the array into a numpy array and transpose it before
        # returning it
        deriv_pol_tensors = np.array(deriv_pol_tensors).T
        return deriv_pol_tensors
