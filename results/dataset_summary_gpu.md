# Datasets used

| split           |   clips |   frames |   ball_present |   occluded |   games |    easy |    hard |
|:----------------|--------:|---------:|---------------:|-----------:|--------:|--------:|--------:|
| synthetic train |       3 |      180 |          0.872 |      0.028 | nan     | nan     | nan     |
| synthetic val   |       1 |       60 |          0.900 |      0.000 | nan     | nan     | nan     |
| synthetic test  |       2 |      120 |          0.917 |      0.050 | nan     | nan     | nan     |
| real train      |       3 |      399 |          0.990 |      0.003 |   1.000 |   0.880 |   0.108 |
| real val        |       1 |      108 |          1.000 |      0.000 |   1.000 |   0.907 |   0.093 |
| real test       |       2 |      339 |          1.000 |      0.000 |   9.000 |   0.938 |   0.062 |

Synthetic: procedural broadcast renderer (§3). Real: TrackNet tennis dataset (Huang et al., 2019), 10 broadcast matches at 1280×720, split by match (fine-tune games 1–8, test games 9–10).
