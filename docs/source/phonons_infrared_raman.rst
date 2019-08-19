Phonon energies, infrared and Raman spectra
-------------------------------------------

The study of the vibrational modes of matter (or phonons) is an important topic
in condensed matter. These phonons are key components in order to understand
thermal or electrical conductivity, heat capacity or sound propagation.

Density Functional Theory, by giving a reliable and unexpensive way of
measuring the energy and forces of a system of interest, is often used to
compute these normal modes. 
Experimentally, they are studied by infrared and Raman spectroscopy. The
intensity of both types of spectroscopy differ for each mode and both can be
studied from first principles.

Three workflows are implemented in MyBigDFT to allow the computation of these
quantities:

* The :class:`~mybigdft.workflows.phonons.Phonons` class allows to compute
  these phonons, and especially their energies.
* The :class:`~mybigdft.workflows.infraredspectrum.InfraredSpectrum` class
  allows to compute the infrared spectrum (that is, measuring the energies and
  their related intensities).
  It requires the same number of DFT calculations used to compute the phonons:
  once the phonons are computed, the infrared intensities can be readily obtained
  from the variations of the dipole moments along these modes. No extra BigDFT
  calculations is therefore required.
* The :class:`~mybigdft.workflows.ramanspectrum.RamanSpectrum` class allows to
  compute the Raman spectrum. It does so at the expense of requiring even more
  DFT calculations, due to the need for the derivatives of the electronic
  polarizability tensor for each normal mode.

To get further details on each workflow, see the links below:

.. toctree::
    :maxdepth: 1

    phonons
    infrared
    raman
