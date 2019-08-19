These attributes allow to get most of the relevant data contained in a Logfile. They provide simple shortcuts instead of having to know exactly where to look for them in the yaml output file, represented by the Logfile class. If some attributes do not appear in the BigDFT output, they default to `None`.

evals
   :Returns: Orbital energies and occupations.

forcemax_cv
   :Returns: Convergence criterion on forces.

kpts
   :Returns: Grid of k-points.

magnetization
   :Returns: Total magnetization of the system.

symmetry
   :Returns: Symmetry group.

energy
   :Returns: Energy (Hartree).

astruct
   :Returns: Atomic structure.

support_functions
   :Returns: Support functions.

electrostatic_multipoles
   :Returns: Electrostatic multipoles.

force_fluct
   :Returns: Threshold fluctuation of Forces.

n_at
   :Returns: Number of Atoms.

dipole
   :Returns: Electric Dipole Moment (AU).

cell
   :Returns: Cell size.

forces
   :Returns: Atomic Forces (Ha/Bohr).

WARNINGS
   :Returns: Warnings raised during the BigDFT run.

kpt_mesh
   :Returns: No. of Monkhorst-Pack grid points.

pressure
   :Returns: Pressure (GPa).

fermi_level
   :Returns: Fermi level.

atom_types
   :Returns: List of the atomic types present in the posinp.

walltime
   :Returns: Walltime since initialization.

boundary_conditions
   :Returns: Boundary Conditions.

forcemax
   :Returns: Maximum value of forces.

sdos
   :Returns: SDos files.

gnrm_cv
   :Returns: Convergence criterion on wavefunction residue.
