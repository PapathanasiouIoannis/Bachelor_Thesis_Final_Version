from framework.eos_catalog import CFL_CATALOG, HADRONIC_CATALOG, literature_catalog_rows
from framework.audit_family_amplitudes import selected_catalog_entries
from src.physics.get_eos_library import get_eos_library


def test_hadronic_catalog_exactly_covers_repository_library():
    core_library, _ = get_eos_library()
    catalog_ids = {entry.eos_id for entry in HADRONIC_CATALOG}

    assert catalog_ids == set(core_library)
    assert len(HADRONIC_CATALOG) == 21
    assert len({entry.family_group_id for entry in HADRONIC_CATALOG}) == 13
    assert all(not entry.exact_formula_primary_verified for entry in HADRONIC_CATALOG)


def test_catalog_records_ps_provenance_and_transition_exception():
    ps = next(entry for entry in HADRONIC_CATALOG if entry.eos_id == "PS")

    assert ps.transition_pressure_mev_fm3 == 0.696
    assert not ps.underlying_primary_url
    assert "no primary citation" in ps.provenance_note.lower()


def test_primary_cfl_table_is_complete_and_contains_controlled_cfl4():
    assert len(CFL_CATALOG) == 19
    assert len({entry.family_group_id for entry in CFL_CATALOG}) == 19
    assert len({entry.parameter_block_id for entry in CFL_CATALOG}) == 5

    cfl4 = next(entry for entry in CFL_CATALOG if entry.eos_id == "CFL4")
    assert (cfl4.bag_b_mev_fm3, cfl4.gap_delta_mev, cfl4.strange_mass_mev) == (
        60,
        100,
        150,
    )
    assert cfl4.published_mmax_msun == 2.127
    assert cfl4.published_r_at_mmax_km == 11.41


def test_combined_machine_readable_catalog_has_40_rows():
    rows = literature_catalog_rows()

    assert len(rows) == 40
    assert {row["matter_class"] for row in rows} == {"hadronic", "quark"}


def test_one_week_profile_selection_is_catalogued_and_excludes_uncited_ps(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        '{"recommended_2p0_eos_ids": ["APR-1", "CFL4"]}', encoding="utf-8"
    )

    entries = selected_catalog_entries(summary_path)

    assert [entry.eos_id for entry in entries] == ["APR-1", "CFL4"]
