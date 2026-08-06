# Constants and Configuration

Fundamental constants and unit conversions remain centralized in `src/config.py` for the thesis-era solver modules. New managed experiments take their user-adjustable EoS, deformation, acceptance, and numerical settings from a validated TOML profile and pass the resolved values explicitly to the scientific workers. Users therefore change a file under `configs/`, not Python source code. The launcher records both the resolved configuration and source hash in each run manifest.

### Geometrized Units
When evaluating the TOV equations, standard nuclear units (like MeV/fm$^3$) result in massive floating-point disparities between the pressure scale and the radius scale, causing numerical stiffness inside the RK45 solver. 

To maintain stability, the entire framework operates on strict geometrized conversions. Using the fundamental constants:
- $M_\odot = 1.989 \times 10^{30} \, \text{kg}$
- $c = 2.9979 \times 10^8 \, \text{m/s}$
- $G = 6.6743 \times 10^{-11} \, \text{m}^3 \text{kg}^{-1} \text{s}^{-2}$

The script establishes exact scaling factors:
- The geometric mass conversion $\approx 1.4766$ km.
- The pressure and density conversion $\approx 1.124 \times 10^{-5}$ km$^{-2}$/MeV.

These multipliers are injected directly into `tov_rhs.py` to seamlessly transition the physics from nuclear space into general relativistic geometry.
