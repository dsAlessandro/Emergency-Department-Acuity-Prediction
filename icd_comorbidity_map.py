"""
ICD-9 / ICD-10 → Triagegeist `hx_*` comorbidity flags

Mapping for the 25 `hx_*` columns defined in schema_target.json.

References
----------
- Quan H et al. (2005). "Coding Algorithms for Defining Comorbidities in
  ICD-9-CM and ICD-10 Administrative Data." Medical Care 43(11):1130-1139.
  (gold-standard adaptation of Charlson and Elixhauser indices)
- AHRQ Clinical Classifications Software Refined (CCSR).
- CDC ICD-9-CM and ICD-10-CM official codebooks.

Assumptions about the input
---------------------------
- MIMIC-IV `diagnoses_icd.icd_code` is stored without the decimal point
  (e.g., "4019" instead of "401.9", "I5032" instead of "I50.32").
- `icd_version` is either 9 or 10.
- All `seq_num` values are used (primary diagnosis + comorbidities).

Usage
-----
    from icd_comorbidity_map import compute_hx_flags
    hx_df = compute_hx_flags(diagnoses_icd_df)
    # hx_df has columns: subject_id (or hadm_id) + 25 hx_* binary columns

How prefix matching works
-------------------------
For most conditions, an ICD code matches if it *starts with* one of the listed
prefixes. Diabetes type 1 vs type 2 in ICD-9 is a special case: both share
the prefix "250", but the 5th digit distinguishes them (0/2 = type 2 or
unspecified, 1/3 = type 1). Handled explicitly in `_diabetes_icd9_type`.
"""

from __future__ import annotations

import pandas as pd

# -----------------------------------------------------------------------------
# Main mapping: for each hx_* flag, list of ICD-9 and ICD-10 code prefixes
# -----------------------------------------------------------------------------
# Codes are stored WITHOUT dots (MIMIC convention).
# A diagnosis matches if its code .startswith() any listed prefix.
# Exceptions (diabetes type 1/2 in ICD-9) are handled in special_rules below.

ICD_COMORBIDITY_MAP: dict[str, dict[str, list[str]]] = {
    # --- Cardiovascular ---
    "hx_hypertension": {
        "icd9":  ["401", "402", "403", "404", "405"],
        "icd10": ["I10", "I11", "I12", "I13", "I15", "I16"],
    },
    "hx_heart_failure": {
        # Includes hypertensive heart disease WITH heart failure
        "icd9":  ["428",
                  "40201", "40211", "40291",
                  "40401", "40403", "40411", "40413", "40491", "40493"],
        "icd10": ["I50", "I110", "I130", "I132"],
    },
    "hx_atrial_fibrillation": {
        "icd9":  ["4273"],            # 427.31 afib, 427.32 aflutter
        "icd10": ["I48"],
    },
    "hx_coronary_artery_disease": {
        "icd9":  ["410", "411", "412", "413", "414", "4292"],
        "icd10": ["I20", "I21", "I22", "I23", "I24", "I25"],
    },
    "hx_stroke_prior": {
        "icd9":  ["430", "431", "432", "433", "434", "435", "436", "437", "438",
                  "V1254"],
        "icd10": ["I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69",
                  "G45", "G46", "Z8673"],
    },
    "hx_peripheral_vascular_disease": {
        "icd9":  ["440", "441", "4431", "4432", "4438", "4439",
                  "4471", "5571", "5579", "V434"],
        "icd10": ["I70", "I71", "I731", "I738", "I739",
                  "I771", "I79", "K551", "K558", "K559", "Z958", "Z959"],
    },

    # --- Metabolic / endocrine ---
    "hx_diabetes_type2": {
        # ICD-9 250 handled by special_rules (5th digit discriminator).
        # Here: include also "E11" unambiguously.
        "icd9":  [],                   # handled in special_rules
        "icd10": ["E11"],
    },
    "hx_diabetes_type1": {
        "icd9":  [],                   # handled in special_rules
        "icd10": ["E10"],
    },
    "hx_obesity": {
        "icd9":  ["2780"],
        "icd10": ["E66"],
    },
    "hx_hypothyroidism": {
        "icd9":  ["243", "244"],
        "icd10": ["E00", "E01", "E02", "E03"],
    },
    "hx_hyperthyroidism": {
        "icd9":  ["242"],
        "icd10": ["E05"],
    },

    # --- Respiratory ---
    "hx_asthma": {
        "icd9":  ["493"],
        "icd10": ["J45", "J46"],
    },
    "hx_copd": {
        "icd9":  ["490", "491", "492", "494", "496"],
        "icd10": ["J40", "J41", "J42", "J43", "J44", "J47"],
    },

    # --- Renal / hepatic ---
    "hx_ckd": {
        "icd9":  ["585", "586",
                  "40301", "40311", "40391",
                  "40402", "40403", "40412", "40413", "40492", "40493",
                  "V420", "V4511", "V56"],
        "icd10": ["N18", "N19", "Z49", "Z940", "Z992"],
    },
    "hx_liver_disease": {
        "icd9":  ["570", "571", "572", "573",
                  "4560", "4561", "4562", "V427"],
        "icd10": ["K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77",
                  "B18", "B19", "Z944"],
    },

    # --- Oncology / immune ---
    "hx_malignancy": {
        # All solid tumors, lymphomas, leukemias, in-situ neoplasms, and
        # personal history of malignancy.
        "icd9":  [str(c) for c in range(140, 209)] + ["V10"],
        "icd10": ["C"] + [f"D0{d}" for d in range(0, 10)] + ["Z85"],
    },
    "hx_hiv": {
        "icd9":  ["042", "V08"],
        "icd10": ["B20", "B21", "B22", "B23", "B24", "Z21"],
    },
    "hx_coagulopathy": {
        "icd9":  ["286", "2871", "2873", "2874", "2875"],
        "icd10": ["D65", "D66", "D67", "D68",
                  "D691", "D693", "D694", "D695", "D696"],
    },
    "hx_immunosuppressed": {
        # Primary immunodeficiencies + long-term drug-induced immunosuppression.
        # Note: approximation — many immunosuppressed patients are only flagged
        # via chemotherapy / transplant status codes that are not included here.
        "icd9":  ["279", "V5863", "V5869"],
        "icd10": ["D80", "D81", "D82", "D83", "D84", "D89",
                  "Z7952", "Z79899", "Z9481"],
    },

    # --- Neuro / psychiatric ---
    "hx_depression": {
        "icd9":  ["2962", "2963", "3004", "311"],
        "icd10": ["F32", "F33", "F341"],
    },
    "hx_anxiety": {
        "icd9":  ["3000", "3002", "3003"],
        "icd10": ["F40", "F41"],
    },
    "hx_dementia": {
        "icd9":  ["290", "2941", "3310", "3311", "3312", "33182"],
        "icd10": ["F00", "F01", "F02", "F03", "G30", "G310", "G3183"],
    },
    "hx_epilepsy": {
        "icd9":  ["345"],
        "icd10": ["G40"],
    },
    "hx_substance_use_disorder": {
        "icd9":  ["291", "292", "303", "304", "305"],
        "icd10": ["F10", "F11", "F12", "F13", "F14", "F15",
                  "F16", "F17", "F18", "F19"],
    },

    # --- Obstetric ---
    "hx_pregnant": {
        "icd9":  [str(c) for c in range(630, 680)] + ["V22", "V23", "V24", "V27", "V28"],
        "icd10": ["O"] + ["Z33", "Z34", "Z36"],
    },
}


# -----------------------------------------------------------------------------
# Special rules: conditions whose mapping cannot be expressed as simple prefixes
# -----------------------------------------------------------------------------

def _diabetes_icd9_type(code: str) -> str | None:
    """
    ICD-9 diabetes (250.xx) 5th character:
      0, 2 → type 2 or unspecified
      1, 3 → type 1

    Returns 'type2', 'type1', or None if not a diabetes code.
    """
    if not isinstance(code, str) or not code.startswith("250"):
        return None
    if len(code) < 5:
        return "type2"  # fallback: unspecified → type 2
    fifth = code[4]
    if fifth in ("0", "2"):
        return "type2"
    if fifth in ("1", "3"):
        return "type1"
    return "type2"


# -----------------------------------------------------------------------------
# Apply mapping to a diagnoses_icd DataFrame
# -----------------------------------------------------------------------------

def compute_hx_flags(
    diagnoses: pd.DataFrame,
    group_by: str = "subject_id",
) -> pd.DataFrame:
    """
    Build the 25 hx_* binary flags from a MIMIC `diagnoses_icd` DataFrame.

    Parameters
    ----------
    diagnoses : pd.DataFrame
        Must contain columns: `icd_code`, `icd_version`, and `group_by`
        (default `subject_id`). You typically pre-filter this to include only
        diagnoses from hadm_ids that occurred BEFORE the target ED visit,
        to avoid leakage.
    group_by : str
        The column to aggregate over. Usually `subject_id` (all history) or
        `hadm_id` (diagnoses of a specific admission).

    Returns
    -------
    pd.DataFrame
        One row per unique value of `group_by`, with 25 hx_* binary columns
        plus `num_comorbidities` (= sum of flags).
    """
    df = diagnoses[[group_by, "icd_code", "icd_version"]].copy()
    df["icd_code"] = df["icd_code"].astype(str).str.strip().str.upper()
    df["icd_version"] = df["icd_version"].astype(int)

    # Initialize one boolean column per hx_* flag, filled at the row level
    flag_cols = list(ICD_COMORBIDITY_MAP.keys())
    for col in flag_cols:
        df[col] = False

    # --- Prefix-based matching ---
    for flag, prefixes in ICD_COMORBIDITY_MAP.items():
        if prefixes["icd9"]:
            mask9 = (df["icd_version"] == 9) & df["icd_code"].str.startswith(
                tuple(prefixes["icd9"])
            )
            df.loc[mask9, flag] = True
        if prefixes["icd10"]:
            mask10 = (df["icd_version"] == 10) & df["icd_code"].str.startswith(
                tuple(prefixes["icd10"])
            )
            df.loc[mask10, flag] = True

    # --- Special rule: ICD-9 diabetes type 1 vs 2 ---
    mask_dm9 = (df["icd_version"] == 9) & df["icd_code"].str.startswith("250")
    dm_types = df.loc[mask_dm9, "icd_code"].apply(_diabetes_icd9_type)
    df.loc[dm_types[dm_types == "type2"].index, "hx_diabetes_type2"] = True
    df.loc[dm_types[dm_types == "type1"].index, "hx_diabetes_type1"] = True

    # --- Aggregate per patient/admission ---
    hx = df.groupby(group_by)[flag_cols].any().astype(int).reset_index()
    hx["num_comorbidities"] = hx[flag_cols].sum(axis=1)

    return hx
