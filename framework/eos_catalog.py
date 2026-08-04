"""Literature and grouping metadata for the repository EoS baselines.

The hadronic formulas in :mod:`src.physics.get_eos_library` are analytic
surrogates reproduced in Stergakis (2025).  Their model-family citations are
primary papers, but those papers have not been shown to publish the exact
surrogate coefficients used by this repository.  Keeping those two provenance
layers separate prevents a local name such as ``APR-1`` from being mistaken
for a canonical tabulation.

The CFL entries are the 19 parameter tuples and reference maximum-star values
published in Table I and Table II of Vasquez Flores & Lugones (2017).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


HADRONIC_FIT_SOURCE_TITLE = (
    "Reconstruction of the Equations of State (EoSs) of Compact Stars using "
    "machine and deep learning regression techniques"
)
HADRONIC_FIT_SOURCE_URL = "https://arxiv.org/abs/2509.13037"
CFL_PRIMARY_SOURCE_TITLE = (
    "Constraining color flavor locked strange stars in the gravitational wave era"
)
CFL_PRIMARY_SOURCE_URL = "https://arxiv.org/abs/1702.02081"


@dataclass(frozen=True)
class HadronicCatalogEntry:
    eos_id: str
    family_group_id: str
    model_family: str
    underlying_primary_title: str
    underlying_primary_url: str
    source_reference_numbers: str
    transition_pressure_mev_fm3: float = 0.184
    exact_formula_primary_verified: bool = False
    provenance_note: str = (
        "Exact repository coefficients verified only against the 2025 fit source; "
        "underlying primary paper identifies the model family."
    )

    def as_row(self) -> dict:
        row = asdict(self)
        row.update(
            {
                "matter_class": "hadronic",
                "parameter_block_id": "",
                "model_superfamily_id": "H_REPOSITORY_SURROGATES",
                "fit_source_title": HADRONIC_FIT_SOURCE_TITLE,
                "fit_source_url": HADRONIC_FIT_SOURCE_URL,
                "bag_b_mev_fm3": None,
                "gap_delta_mev": None,
                "strange_mass_mev": None,
                "published_mmax_msun": None,
                "published_r_at_mmax_km": None,
            }
        )
        return row


@dataclass(frozen=True)
class CflCatalogEntry:
    eos_id: str
    bag_b_mev_fm3: float
    gap_delta_mev: float
    strange_mass_mev: float
    published_mmax_msun: float
    published_r_at_mmax_km: float

    @property
    def family_group_id(self) -> str:
        # Every fixed microphysical tuple is one baseline family. All Gaussian
        # amplitudes derived from this tuple must remain in the same ML split.
        return f"Q_{self.eos_id}"

    @property
    def parameter_block_id(self) -> str:
        # A bag-constant block supplies a stricter secondary OOD grouping.
        return f"Q_CFL_B{int(self.bag_b_mev_fm3):03d}"

    def as_row(self) -> dict:
        return {
            "matter_class": "quark",
            "eos_id": self.eos_id,
            "family_group_id": self.family_group_id,
            "parameter_block_id": self.parameter_block_id,
            "model_superfamily_id": "Q_ANALYTIC_CFL_MIT_BAG",
            "model_family": "analytic CFL MIT-bag",
            "underlying_primary_title": CFL_PRIMARY_SOURCE_TITLE,
            "underlying_primary_url": CFL_PRIMARY_SOURCE_URL,
            "source_reference_numbers": "Vasquez Flores & Lugones (2017), Tables I-II",
            "transition_pressure_mev_fm3": 0.0,
            "exact_formula_primary_verified": True,
            "provenance_note": (
                "Tuple and reference maximum-star values are published in the "
                "primary source; repository uses its O(m_s^2, Delta^2) analytic EoS."
            ),
            "fit_source_title": CFL_PRIMARY_SOURCE_TITLE,
            "fit_source_url": CFL_PRIMARY_SOURCE_URL,
            "bag_b_mev_fm3": self.bag_b_mev_fm3,
            "gap_delta_mev": self.gap_delta_mev,
            "strange_mass_mev": self.strange_mass_mev,
            "published_mmax_msun": self.published_mmax_msun,
            "published_r_at_mmax_km": self.published_r_at_mmax_km,
        }


_PRIMARY = {
    "APR": (
        "Equation of state of nucleon matter and neutron star structure",
        "https://arxiv.org/abs/nucl-th/9804027",
        "Stergakis ref. 9",
    ),
    "BGP": (
        "Relativistic superdense matter in cold systems: Theory",
        "https://doi.org/10.1103/PhysRevD.12.3043",
        "Stergakis ref. 10",
    ),
    "BL": (
        "Equation of state of dense nuclear matter and neutron star structure "
        "from nuclear chiral interactions",
        "https://arxiv.org/abs/1805.11846",
        "Stergakis ref. 11",
    ),
    "DH": (
        "A unified equation of state of dense matter and neutron star structure",
        "https://arxiv.org/abs/astro-ph/0111092",
        "Stergakis ref. 12",
    ),
    "HHJ": (
        "Phases of dense matter in neutron stars",
        "https://arxiv.org/abs/nucl-th/9902033",
        "Stergakis ref. 13",
    ),
    "HLPS": (
        "Equation of state and neutron star properties constrained by nuclear "
        "physics and observation",
        "https://arxiv.org/abs/1303.4662",
        "Stergakis ref. 14",
    ),
    "MDI": (
        "Composition and structure of protoneutron stars; Equation of state for "
        "beta-stable hot nuclear matter",
        "https://arxiv.org/abs/0805.0353",
        "Stergakis refs. 15-16",
    ),
    "NLD": (
        "Momentum dependent mean-field dynamics of compressed nuclear matter and "
        "neutron stars; Toward relativistic mean-field description of N-nucleus reactions",
        "https://doi.org/10.1016/j.nuclphysa.2013.01.002",
        "Stergakis refs. 17-18",
    ),
    "SCVBB": (
        "Unified equation of state for neutron stars on a microscopic basis",
        "https://arxiv.org/abs/1506.00375",
        "Stergakis ref. 19",
    ),
    "SKYRME": (
        "A Skyrme parametrization from subnuclear to neutron star densities; "
        "Nuclear-matter incompressibility from fits of generalized Skyrme force "
        "to breathing-mode energies",
        "https://doi.org/10.1016/S0375-9474(97)00596-4",
        "Stergakis refs. 20-21",
    ),
    "WALECKA": (
        "A theory of highly-condensed matter",
        "https://doi.org/10.1016/0003-4916(74)90132-8",
        "Stergakis ref. 22",
    ),
    "WFF": (
        "Equation of state for dense nucleon matter",
        "https://doi.org/10.1103/PhysRevC.38.1010",
        "Stergakis ref. 23",
    ),
}


def _hadronic(
    eos_id: str,
    family_group_id: str,
    model_family: str,
    primary_key: str | None,
    *,
    transition_pressure: float = 0.184,
) -> HadronicCatalogEntry:
    if primary_key is None:
        return HadronicCatalogEntry(
            eos_id=eos_id,
            family_group_id=family_group_id,
            model_family=model_family,
            underlying_primary_title="",
            underlying_primary_url="",
            source_reference_numbers="No primary citation attached to PS in fit source",
            transition_pressure_mev_fm3=transition_pressure,
            provenance_note=(
                "Exact repository coefficients verified against the 2025 fit source, "
                "which gives no primary citation for PS."
            ),
        )
    title, url, references = _PRIMARY[primary_key]
    return HadronicCatalogEntry(
        eos_id=eos_id,
        family_group_id=family_group_id,
        model_family=model_family,
        underlying_primary_title=title,
        underlying_primary_url=url,
        source_reference_numbers=references,
        transition_pressure_mev_fm3=transition_pressure,
    )


HADRONIC_CATALOG = (
    _hadronic("MDI-1", "H_MDI", "Momentum-Dependent Interaction", "MDI"),
    _hadronic("MDI-2", "H_MDI", "Momentum-Dependent Interaction", "MDI"),
    _hadronic("MDI-3", "H_MDI", "Momentum-Dependent Interaction", "MDI"),
    _hadronic("MDI-4", "H_MDI", "Momentum-Dependent Interaction", "MDI"),
    _hadronic("NLD", "H_NLD", "Non-Linear Derivative", "NLD"),
    _hadronic("HHJ-1", "H_HHJ", "Heiselberg-Hjorth-Jensen", "HHJ"),
    _hadronic("HHJ-2", "H_HHJ", "Heiselberg-Hjorth-Jensen", "HHJ"),
    _hadronic("Ska", "H_SKYRME", "Skyrme", "SKYRME"),
    _hadronic("SkI4", "H_SKYRME", "Skyrme", "SKYRME"),
    _hadronic("HLPS-2", "H_HLPS", "Hebeler-Lattimer-Pethick-Schwenk", "HLPS"),
    _hadronic("HLPS-3", "H_HLPS", "Hebeler-Lattimer-Pethick-Schwenk", "HLPS"),
    _hadronic("SCVBB", "H_SCVBB", "Sharma-Centelles-Vinas-Baldo-Burgio", "SCVBB"),
    _hadronic("WFF-1", "H_WFF", "Wiringa-Fiks-Fabrocini", "WFF"),
    _hadronic("WFF-2", "H_WFF", "Wiringa-Fiks-Fabrocini", "WFF"),
    _hadronic("PS", "H_PS", "Pethick-Schwenk", None, transition_pressure=0.696),
    _hadronic("W", "H_WALECKA", "Walecka", "WALECKA"),
    _hadronic("BGP", "H_BGP", "Bowers-Gleeson-Pedigo", "BGP"),
    _hadronic("BL-1", "H_BL", "Bombaci-Logoteta", "BL"),
    _hadronic("BL-2", "H_BL", "Bombaci-Logoteta", "BL"),
    _hadronic("DH", "H_DH", "Douchin-Haensel", "DH"),
    _hadronic("APR-1", "H_APR", "Akmal-Pandharipande-Ravenhall", "APR"),
)


CFL_CATALOG = (
    CflCatalogEntry("CFL1", 60, 50, 0, 2.051, 11.08),
    CflCatalogEntry("CFL2", 60, 50, 150, 1.830, 10.09),
    CflCatalogEntry("CFL3", 60, 100, 0, 2.357, 12.38),
    CflCatalogEntry("CFL4", 60, 100, 150, 2.127, 11.41),
    CflCatalogEntry("CFL5", 60, 150, 0, 2.842, 14.24),
    CflCatalogEntry("CFL6", 60, 150, 150, 2.631, 13.46),
    CflCatalogEntry("CFL7", 80, 100, 0, 1.994, 10.52),
    CflCatalogEntry("CFL8", 80, 100, 150, 1.821, 9.79),
    CflCatalogEntry("CFL9", 80, 150, 0, 2.365, 11.98),
    CflCatalogEntry("CFL10", 80, 150, 150, 2.202, 11.36),
    CflCatalogEntry("CFL11", 100, 50, 0, 1.571, 8.51),
    CflCatalogEntry("CFL12", 100, 100, 0, 1.754, 9.29),
    CflCatalogEntry("CFL13", 100, 100, 150, 1.616, 8.70),
    CflCatalogEntry("CFL14", 100, 150, 0, 2.055, 10.49),
    CflCatalogEntry("CFL15", 100, 150, 150, 1.922, 9.98),
    CflCatalogEntry("CFL16", 120, 100, 0, 1.582, 8.40),
    CflCatalogEntry("CFL17", 120, 150, 0, 1.834, 9.42),
    CflCatalogEntry("CFL18", 120, 150, 150, 1.722, 8.98),
    CflCatalogEntry("CFL19", 140, 150, 0, 1.667, 8.60),
)


def literature_catalog_rows() -> list[dict]:
    """Return combined catalog rows suitable for CSV or JSON serialization."""

    return [
        *(entry.as_row() for entry in HADRONIC_CATALOG),
        *(entry.as_row() for entry in CFL_CATALOG),
    ]
