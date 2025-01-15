FEniCS Code for Brain Tissue Torsion Simulation
This repository contains the FEniCS code for finite element simulations of the torsion of a solid brain cylinder using the Modified Quasi-Linear Viscoelastic (MQLV) model. These simulations are designed to characterise the viscoelastic properties of brain tissue based on experimental data.

Features
Implements the MQLV model to simulate brain tissue deformation under torsion.
Includes pre-compression and torsion steps with detailed boundary conditions.
Generates displacement and Cauchy stress tensor components for analysis.
Outputs simulation data for visualisation in ParaView.

Prerequisites
FEniCS and Python installed 
Recommended: ParaView for visualising results

Running the Simulation
Open the main script file, cylinder_viscoelasticity.py, and configure the parameters (e.g., material properties, geometry, and time steps).
Run the script
The results (e.g., displacement and stress tensor data) will be saved in the specified output directory.

Visualising Results
Open the generated files in ParaView for 3D visualisation and analysis.

Repository Structure
cylinder_viscoelasticity.py: Main driver for the simulations.
README.md: Documentation.
License

Acknowledgements
This work is based on research conducted by Griffen Small, Francesca Ballatore, Chiara Giverso and Valentina Balbi (School of Mathematical and Statistical Sciences, University of Galway and Department of Mathematical Sciences G. L. Lagrange, Politecnico di Torino). For more details, refer to the associated publications.
