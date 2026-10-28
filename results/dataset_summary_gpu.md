# Datasets used

| split           |   clips |   frames |   ball_present |   occluded | games                  |    easy |    hard |
|:----------------|--------:|---------:|---------------:|-----------:|:-----------------------|--------:|--------:|
| synthetic train |     150 |     9000 |          0.893 |      0.050 | nan                    | nan     | nan     |
| synthetic val   |      12 |      720 |          0.914 |      0.039 | nan                    | nan     | nan     |
| synthetic test  |      30 |     1800 |          0.884 |      0.047 | nan                    | nan     | nan     |
| real train      |      66 |    13740 |          0.963 |      0.005 | 1, 2, 3, 4, 5, 6, 7, 8 |   0.890 |   0.068 |
| real val        |       8 |     2307 |          0.951 |      0.005 | 1, 2, 3, 4, 5, 6, 7, 8 |   0.868 |   0.078 |
| real test       |      21 |     3693 |          0.976 |      0.001 | 9, 10                  |   0.900 |   0.075 |

Synthetic: procedural broadcast renderer (§3). Real: TrackNet tennis dataset (Huang et al., 2019), 10 broadcast matches at 1280×720, split by match (fine-tune games 1–8, test games 9–10).
