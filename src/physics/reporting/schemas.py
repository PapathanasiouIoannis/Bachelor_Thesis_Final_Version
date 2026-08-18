"""Column schemas and display headings for experiment reporting."""

EOS_COLUMNS = (
    "matter_type",
    "baseline_name",
    "model_identifier",
    "sweep_id",
    "deformation_amplitude",
    "pair_accepted",
    "eos_validation_passed",
    "eos_validation_reason",
    "eos_region",
    "energy_density_mev_fm3",
    "pressure_mev_fm3",
    "sound_speed_squared",
    "causal_prefix_applied",
    "discarded_suffix_points",
    "first_discarded_sound_speed_squared",
    "causal_cutoff_pressure_mev_fm3",
    "causal_cutoff_energy_density_mev_fm3",
)

STELLAR_COLUMNS = (
    "matter_type",
    "baseline_name",
    "model_identifier",
    "sweep_id",
    "curve_id",
    "deformation_amplitude",
    "mass_msun",
    "radius_km",
    "tidal_deformability",
    "central_pressure_mev_fm3",
    "central_energy_density_mev_fm3",
    "central_sound_speed_squared",
    "surface_energy_density_mev_fm3",
)

_STELLAR_NUMERIC_COLUMNS = (
    "deformation_amplitude",
    "mass_msun",
    "radius_km",
    "tidal_deformability",
    "central_pressure_mev_fm3",
    "central_energy_density_mev_fm3",
    "central_sound_speed_squared",
    "surface_energy_density_mev_fm3",
)

SUMMARY_COLUMNS = (
    "matter_type",
    "baseline_name",
    "sweep_id",
    "deformation_amplitude",
    "maximum_mass_msun",
    "radius_1p4_km",
    "tidal_deformability_1p4",
    "turning_point_stability_estimate",
    "status",
)

SUMMARY_HEADINGS = {
    "matter_type": "Matter type",
    "baseline_name": "EoS baseline",
    "sweep_id": "Sweep identifier",
    "deformation_amplitude": "Amplitude A",
    "maximum_mass_msun": r"Maximum mass [$M_\odot$]",
    "radius_1p4_km": r"Radius at $1.4M_\odot$ [km]",
    "tidal_deformability_1p4": r"Tidal deformability at $1.4M_\odot$",
    "turning_point_stability_estimate": "Turning-point stability estimate",
    "status": "Status",
}

REJECTION_HEADINGS = {
    "sweep_id": "Sweep identifier",
    "deformation_amplitude": "Amplitude A",
    "matter_type": "Failed matter type",
    "stage": "Stage",
    "exception_type": "Error type",
    "reason": "Reason",
}

CONVERGENCE_HEADINGS = {
    "matter_type": "Matter type",
    "baseline_name": "EoS baseline",
    "deformation_amplitude": "Amplitude A",
    "check": "Refinement",
    "delta_maximum_mass_msun": r"Absolute maximum-mass change [$M_\odot$]",
    "delta_radius_1p4_km": r"Absolute radius change at $1.4M_\odot$ [km]",
    "relative_delta_tidal_deformability_1p4": (
        "Relative tidal-deformability change at 1.4 M_sun"
    ),
    "maximum_mass_passed": "Maximum-mass threshold passed",
    "radius_1p4_passed": "Radius threshold passed",
    "tidal_deformability_1p4_passed": "Tidal threshold passed",
    "refined_physical_requirements_passed": ("Refined physical requirements passed"),
    "refined_physical_requirements_reason": "Refined physical-requirement result",
    "passed": "All thresholds passed",
}

CAUSAL_DOMAIN_COLUMNS = (
    "matter_type",
    "baseline_name",
    "model_identifier",
    "sweep_id",
    "deformation_amplitude",
    "pair_accepted",
    "eos_validation_passed",
    "eos_validation_reason",
    "causal_prefix_applied",
    "discarded_suffix_points",
    "first_discarded_sound_speed_squared",
    "causal_cutoff_pressure_mev_fm3",
    "causal_cutoff_energy_density_mev_fm3",
)

CAUSAL_DOMAIN_HEADINGS = {
    "matter_type": "Matter type",
    "baseline_name": "EoS baseline",
    "model_identifier": "Parameterized model identifier",
    "sweep_id": "Sweep identifier",
    "deformation_amplitude": "Amplitude A",
    "pair_accepted": "Pair accepted",
    "eos_validation_passed": "Complete-table validation passed",
    "eos_validation_reason": "Validation result",
    "causal_prefix_applied": "Causal prefix applied",
    "discarded_suffix_points": "Discarded suffix points",
    "first_discarded_sound_speed_squared": r"First discarded $c_s^2$",
    "causal_cutoff_pressure_mev_fm3": (r"Causal cutoff pressure [MeV fm$^{-3}$]"),
    "causal_cutoff_energy_density_mev_fm3": (
        r"Causal cutoff energy density [MeV fm$^{-3}$]"
    ),
}


__all__ = [
    "CAUSAL_DOMAIN_COLUMNS",
    "CAUSAL_DOMAIN_HEADINGS",
    "CONVERGENCE_HEADINGS",
    "EOS_COLUMNS",
    "REJECTION_HEADINGS",
    "STELLAR_COLUMNS",
    "SUMMARY_COLUMNS",
    "SUMMARY_HEADINGS",
]
