# Data Layout

## Raw Benchmarks

Each benchmark dataset now lives in its own directory under `data/raw/benchmarks/`.

- `abalone/`
- `auto_mpg/`
- `communities_and_crime/`
- `computer_hardware/`
- `concrete_compressive_strength/`
- `concrete_slump_test/`
- `parkinsons_telemonitoring/`

Each folder keeps the raw data file, any names/readme metadata, and the original archive when it exists.

`concrete_compressive_strength/` is preserved as a raw source directory, but it is not yet wired into the current loader registry because it depends on Excel ingestion.

## Processed Data

- `data/processed/dat_clean.rda`: existing processed artifact preserved from the original workspace

## Next Cleanup Target

The next step for the real-data benchmark is to add canonical loader code for each dataset and document:

- target column
- dropped identifier columns
- missing-value handling
- train/validation/calibration/test split policy
