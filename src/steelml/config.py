"""Project-wide paths and constants.

Every pipeline script imports these values so that a choice (a seed, a
cooling rate, an element list) is made in exactly one place.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"  # large downloads, not committed
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

RANDOM_SEED = 42

# Elements tracked everywhere (wt%). Fe is the balance.
ELEMENTS = ["C", "Mn", "Si", "P", "S", "Cr", "Ni", "Mo", "V", "Cu",
            "Al", "Ti", "Nb", "N", "W", "Co", "B"]

# Typical residual levels (wt%) assumed when a specification is silent about
# an element. They describe ordinary electric-arc/BOF practice and are used
# only to fill unspecified residuals when compositions are sampled.
RESIDUAL_TYPICAL = {"Cr": 0.10, "Ni": 0.10, "Mo": 0.03, "Cu": 0.15,
                    "V": 0.005, "Al": 0.03, "Ti": 0.002, "Nb": 0.002,
                    "N": 0.008, "W": 0.0, "Co": 0.0, "B": 0.0,
                    "P": 0.015, "S": 0.015, "Si": 0.25}

# Cooling-route conventions -------------------------------------------------
# Furnace cooling (full anneal): the charge cools inside the switched-off (or
# programme-controlled) furnace. Recommended full-annealing practice cools
# through the transformation range at roughly 10-50 K/h, slower for the
# sluggish alloy steels. Nominal: 0.005 K/s (18 K/h) at 700 C, which is also
# the value that best reconciles the transformation model with the handbook
# annealed hardnesses (pipeline/s03_transformation_model.py). Sensitivity
# runs cover 0.003-0.03 K/s.
FC_RATE_700C = 0.005         # K/s at 700 C (nominal)
FC_RATE_RANGE = (0.003, 0.03)

# Air cooling (normalizing): still air, rate set by section size through a
# lumped-capacitance balance (convection + radiation), see physics.cooling.
AC_DEFAULT_DIAMETER_MM = 25.0  # default for what-if studies (single 25 mm bar, ~1 K/s)
AC_DIAMETER_RANGE_MM = (10.0, 100.0)

# Effective conditions of the ASM/Bethlehem handbook data, calibrated in
# s03 from a scan of the physics-baseline hardness error (no ML involved).
# Normalized rows: the error is flat between ~50 and 80 mm bars in still air
# (21-22 HB RMSE vs 58 HB at 25 mm); 50 mm (~0.5 K/s at 700 C, e.g. bars
# cooled in a rack rather than singly) is taken at the start of that flat
# region. Annealed rows: the error hardly changes below 0.01 K/s; 0.005 K/s
# (18 K/h) is the usual full-annealing furnace cool for alloy steels.
HANDBOOK_AC_SECTION_MM = 50.0
HANDBOOK_FC_RATE = 0.005

# Transformation-model constants (multipliers on the Li et al. rate constants
# and a shift of their bainite-start temperature; see physics.kinetics).
# * Ferrite and pearlite: the published values (1.0). A scan in stage 3 shows
#   that a faster ferrite would fit the measured air-cooled phase fractions but
#   spoil the measured ferrite-start temperatures, and the Bayesian calibration
#   on the SAE H-bands leaves both at 1.0 (the bands do not constrain them).
# * Bainite: calibrated (stage 8). Against the SAE/ASTM Jominy hardenability
#   bands of the non-boron H-steels (about 1000 band points of 65 grades), the
#   published constants are right at the quenched end but too hardenable from
#   J5 on (+12 HRC at J18-J32). The maximum a posteriori calibration lowers
#   the bainite start by 107 C (Laplace sd 6 C) and speeds the reaction up
#   once below it (ln factor 4.37, sd 2.2). Grade-grouped cross-validation:
#   H-band RMSE 10.1 -> 5.9 HRC. It transfers to data it never saw: the
#   dilatometry bainite-start bias goes from +67 to -20 C, Ms from 10.3 to
#   7.4 C, the CCT-atlas hardness from 83 to 69 HV and the handbook hardness
#   of a 25 mm normalised bar from 58 to 31 HB; it does not hold for the more
#   highly alloyed SA-540 (3.5Ni-1.7Cr-0.5Mo), which it makes too bainitic at
#   0.5-1 K/s. Set the two values to 1.0 and 0.0 for the published model.
FERRITE_RATE_FACTOR = 1.0
PEARLITE_RATE_FACTOR = 1.0
BAINITE_RATE_FACTOR = 79.4
BAINITE_START_SHIFT_C = -107.2
# * Pearlite spacing S0 = A * dT^(-m), dT = Ae1 - mean pearlite formation
#   temperature from the kinetic model. Fitted on 14 measured continuous-
#   cooling spacings from 4 papers (leave-one-source-out typical error x1.7).
#   The exponent below 1 compensates for the kinetic model forming air-cooled
#   pearlite too cold; with it, air-cooled spacings match measurements and
#   furnace-cooled ones land at textbook coarse-pearlite values (0.5-0.9 um),
#   which a constant k cannot do (k = 8.02 um K, isothermal Marder-Bramfitt,
#   is ~3x too fine for air cooling; a k fitted to air cooling makes furnace-
#   cooled pearlite 2-3 um).
SPACING_A_UM = 3.33
SPACING_EXPONENT = 0.607

# Default prior-austenite grain size (ASTM number) for fine-grained,
# aluminium-killed steels austenitised at 800-925 C.
DEFAULT_ASTM_G = 7.5

PSI_TO_MPA = 0.00689475729


def ensure_dirs():
    for d in (RAW_DIR, EXTERNAL_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
