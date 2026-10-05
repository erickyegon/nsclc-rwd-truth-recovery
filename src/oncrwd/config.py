"""Central configuration: paths, study design constants, analysis parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"          # Synthea-style CSVs (what an analyst would receive)
TRUTH_DIR = ROOT / "data" / "truth"      # simulation ground truth, used ONLY for validation
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

SEED = 2026
N_PATIENTS = 6000

# Study design
DATA_CUTOFF = "2024-12-31"        # administrative end of follow-up
IO_ERA_START = "2018-06-01"       # first-line PD-1 + chemo available from this date in the simulation
MAX_DX_TO_INDEX_DAYS = 90         # first-line therapy must start within 90 d of diagnosis

# Line-of-therapy algorithm defaults (see lot.py)
LOT_WINDOW_DAYS = 28              # agents started within this window of line start = same regimen
LOT_GAP_DAYS = 90                 # treatment-free interval that starts a new line
DAYS_PER_MONTH = 30.4375

# Ground-truth log-hazard ratios used by the generator (overall survival, from first-line start)
TRUE_LOG_HR = {
    "io_chemo_vs_chemo": -0.35,
    "io_mono_vs_chemo": -0.30,
    "osimertinib_vs_chemo_egfr": -0.90,
    "erlotinib_vs_chemo_egfr": -0.50,
    "age_per_year": 0.025,
    "ecog_per_level": 0.30,
    "stage4_vs_3": 0.45,
    "brain_mets": 0.30,
    "comorbidity_per_condition": 0.10,
    "male": 0.15,
    "squamous": 0.15,
    "egfr_positive": -0.15,
    "second_line_receipt": 0.0,    # NULL by construction -> used for the immortal-time-bias demo
}
