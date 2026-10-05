"""Unit tests for the line-of-therapy rules, using hand-built administration histories."""
import pandas as pd
import pytest

from oncrwd.drugs import lot_agent, normalise_drug, regimen_class
from oncrwd.lot import build_admin_table, derive_lines


def adm(rows):
    df = pd.DataFrame(rows, columns=["patient_id", "agent", "start", "end"])
    df["start"], df["end"] = pd.to_datetime(df["start"]), pd.to_datetime(df["end"])
    return df


def d(offset):
    return (pd.Timestamp("2020-01-01") + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")


def test_combination_started_together_is_one_line():
    a = adm([("p", "platinum", d(0), d(0)), ("p", "pemetrexed", d(0), d(0)), ("p", "pembrolizumab", d(0), d(0))])
    out = derive_lines(a)
    assert len(out) == 1 and out.loc[0, "regimen"] == "pembrolizumab+pemetrexed+platinum"


def test_platinum_swap_is_not_new_line():
    a = adm([("p", "platinum", d(0), d(0)), ("p", "pemetrexed", d(0), d(0)),
             ("p", "platinum", d(21), d(21)), ("p", "pemetrexed", d(21), d(21))])
    assert len(derive_lines(a)) == 1
    assert lot_agent("cisplatin") == lot_agent("carboplatin") == "platinum"


def test_dropping_agent_into_maintenance_is_not_new_line():
    rows = [("p", "platinum", d(0), d(0)), ("p", "pemetrexed", d(0), d(0)), ("p", "pembrolizumab", d(0), d(0))]
    rows += [("p", x, d(84 + 21 * i), d(84 + 21 * i)) for i in range(4) for x in ("pemetrexed", "pembrolizumab")]
    assert len(derive_lines(adm(rows))) == 1


def test_new_agent_after_window_starts_new_line():
    a = adm([("p", "platinum", d(0), d(0)), ("p", "paclitaxel", d(0), d(0)), ("p", "docetaxel", d(120), d(120))])
    out = derive_lines(a)
    assert list(out["line"]) == [1, 2] and out.loc[1, "regimen"] == "docetaxel"


def test_late_added_agent_inside_window_joins_regimen_and_outside_splits():
    rows = [("p", "platinum", d(0), d(0)), ("p", "pemetrexed", d(0), d(0)), ("p", "pembrolizumab", d(21), d(21))]
    assert len(derive_lines(adm(rows), window=28)) == 1
    rows[-1] = ("p", "pembrolizumab", d(42), d(42))
    assert len(derive_lines(adm(rows), window=28)) == 2        # documented limitation
    assert len(derive_lines(adm(rows), window=56)) == 1


def test_restart_after_long_gap_is_new_line_but_not_after_short_gap():
    base = [("p", "pembrolizumab", d(0), d(0)), ("p", "pembrolizumab", d(21), d(21))]
    assert len(derive_lines(adm(base + [("p", "pembrolizumab", d(60), d(60))]))) == 1
    assert len(derive_lines(adm(base + [("p", "pembrolizumab", d(200), d(200))]), gap=90)) == 2


def test_oral_therapy_supply_window_counts_toward_gap():
    # 30-day supplies back to back leave no gap even though start dates are 30 d apart
    a = adm([("p", "osimertinib", d(0), d(30)), ("p", "osimertinib", d(30), d(60)), ("p", "osimertinib", d(180), d(210))])
    assert len(derive_lines(a, gap=90)) == 2                    # 120-day hole > 90
    assert len(derive_lines(a, gap=150)) == 1


def test_patients_are_independent():
    a = adm([("a", "docetaxel", d(0), d(0)), ("b", "pembrolizumab", d(5), d(5))])
    out = derive_lines(a)
    assert set(out["patient_id"]) == {"a", "b"} and (out["line"] == 1).all()


@pytest.mark.parametrize("text,expected", [
    ("KEYTRUDA", "pembrolizumab"), ("Carboplatin 10 MG/ML Injectable Solution", "carboplatin"),
    ("pemetrexed disodium", "pemetrexed"), ("TAGRISSO", "osimertinib"), ("Dexamethasone 4 MG Oral Tablet", None),
    ("Pegfilgrastim 6 MG/0.6ML Prefilled Syringe", None), (float("nan"), None)])
def test_drug_normalisation(text, expected):
    assert normalise_drug(text) == expected


def test_supportive_meds_are_ignored_when_building_admin_table():
    meds = pd.DataFrame({"PATIENT": ["p", "p"], "DESCRIPTION": ["Ondansetron 8 MG Oral Tablet", "ALIMTA"],
                         "START": ["2020-01-01", "2020-01-01"], "STOP": [None, "2020-01-01"]})
    assert list(build_admin_table(meds)["agent"]) == ["pemetrexed"]


def test_regimen_classes():
    assert regimen_class("pembrolizumab+pemetrexed+platinum") == "IO + chemo"
    assert regimen_class("pembrolizumab") == "IO mono"
    assert regimen_class("paclitaxel+platinum") == "Platinum chemo"
    assert regimen_class("osimertinib") == "EGFR TKI"
    assert regimen_class("docetaxel") == "Other"
