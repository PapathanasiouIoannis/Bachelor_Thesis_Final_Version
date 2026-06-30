# The TOV Equations (src/physics/tov_rhs.py)

In this project, everything starts with the Tolman-Oppenheimer-Volkoff (TOV) equations. These are the fundamental equations of general relativity that describe the structure of a spherically symmetric body of isotropic material in static gravitational equilibrium (Tolman, 1939; Oppenheimer & Volkoff, 1939).

When I wrote 	ov_rhs.py, the goal was to take a given Equation of State (which gives us Pressure $ for a given Energy Density $\epsilon$) and figure out the Mass $, Radius $, and Tidal Deformability $\Lambda$ of the resulting star.

The script sets up the differential equations for pressure, mass, and the tidal metric $. We integrate these equations outward from the center of the star (where  pprox 0$) until the pressure hits zero, which signifies the surface of the star!
