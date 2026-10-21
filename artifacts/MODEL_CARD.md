# Bike Demand Lab · model card

Source: **UCI Bike Sharing · official hour.csv**. Task: Retrospective hourly demand prediction using observed weather.

Selected by validation MAE: **Gradient boosting · 15 leaves**. Scores below are computed after refitting each frozen family winner on train + validation.

| Model | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ |
|---|---:|---:|---:|
| Training mean | 170.509 | 226.027 | -0.078 |
| Hour × workday mean | 94.780 | 138.527 | 0.595 |
| Ridge α=1 | 97.530 | 132.584 | 0.629 |
| Gradient boosting · 15 leaves | 54.896 | 80.970 | 0.862 |

## Evaluation design

- train: 11,539 rows, 2011-01-01T00:00:00 to 2012-04-30T23:00:00.
- validation: 2,952 rows, 2012-05-01T00:00:00 to 2012-08-31T23:00:00.
- test: 2,888 rows, 2012-09-01T00:00:00 to 2012-12-31T23:00:00.

No random split. All transformations fit within their training partition. The fixed grid uses validation MAE; test scores do not choose hyperparameters. The final model uses all train and validation rows, never test rows. This is a single historical holdout, not rolling-origin validation.

## Inputs and target

Target: total hourly rentals (`cnt`). Inputs: calendar fields, a deterministic date trend and recorded weather. `casual` and `registered` sum to the target and are excluded, along with the record ID, `instant`. `yr` is excluded because date already provides the time trend. The feature allowlist applies during training and inference. Numeric median imputation and scaling plus categorical most-frequent imputation and one-hot encoding are learned in the pipeline. No missing target hours are invented.

## Intended use and limitations

A reproducible educational benchmark of historical demand in Washington, D.C., 2011–2012. Weather is observed, so these scores do not establish future forecast quality. Deployment would need forecast-available weather, rolling temporal evaluation, updated local data, and uncertainty estimates. Demand growth, missing hours and unusual events can shift errors. Aggregate rental counts do not describe station availability. No external, university or production validation is claimed.

Permutation importance is the increase in held-out MAE after shuffling one raw input, averaged over five seeded repeats on at most 1,000 test rows. Correlated calendar and weather fields share information; shuffling can create unrealistic combinations. Standard deviations describe repeat variability, not confidence intervals. This diagnostic did not influence model choice.

## Reproducibility and attribution

Seed: 42; CPU thread limit: 2. Data SHA-256: `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`. See `benchmark.json` for versions, all validation candidates, exact boundaries and metrics.

UCI Bike Sharing dataset, Hadi Fanaee-T (2013), [DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894), CC BY 4.0. Fanaee-T and Gama, *Event labeling combining ensemble detectors and background knowledge*, DOI 10.1007/s13748-013-0040-3. See `data/PROVENANCE.md`. For synthetic demo runs, this citation identifies the schema inspiration only; their rows and scores are synthetic.
