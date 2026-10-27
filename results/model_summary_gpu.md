# Layer-by-layer summary (512×288)

## V5 (faithful)

| block   | role                            | input       | output      |   params |
|:--------|:--------------------------------|:------------|:------------|---------:|
| MDD     | motion decoupling               | 9×288×512   | 2×2×288×512 |        2 |
| e1      | encoder 1 (conv ×2)             | 13×288×512  | 32×288×512  |    13088 |
| e2      | encoder 2 (conv ×2)             | 32×144×256  | 64×144×256  |    55552 |
| e3      | encoder 3 (conv ×3)             | 64×72×128   | 128×72×128  |   369408 |
| e4      | bottleneck (conv ×3)            | 128×36×64   | 256×36×64   |  1476096 |
| d3      | decoder 3 (conv ×3)             | 384×72×128  | 128×72×128  |   738048 |
| d2      | decoder 2 (conv ×2)             | 192×144×256 | 64×144×256  |   147712 |
| d1      | decoder 1 (conv ×2)             | 96×288×512  | 32×288×512  |    36992 |
| head    | 1×1 conv → draft (3 maps)       | 32×288×512  | 3×288×512   |       99 |
| R-STR   | space-time Transformer residual | 3×288×512   | 3×288×512   |   260672 |

Total parameters: 3,097,669

## V5-Lite+ (M1–M4)

| block   | role                            | input       | output      |   params |
|:--------|:--------------------------------|:------------|:------------|---------:|
| MDD     | motion decoupling               | 9×288×512   | 2×2×288×512 |        8 |
| e1      | encoder 1 (conv ×2)             | 15×288×512  | 32×288×512  |    13664 |
| e2      | encoder 2 (conv ×2)             | 32×144×256  | 64×144×256  |     7456 |
| e3      | encoder 3 (conv ×3)             | 64×72×128   | 128×72×128  |    45248 |
| e4      | bottleneck (conv ×3)            | 128×36×64   | 256×36×64   |   172416 |
| d3      | decoder 3 (conv ×3)             | 384×72×128  | 128×72×128  |    89728 |
| d2      | decoder 2 (conv ×2)             | 192×144×256 | 64×144×256  |    19456 |
| d1      | decoder 1 (conv ×2)             | 96×288×512  | 32×288×512  |     5632 |
| head    | 1×1 conv → draft (3 maps)       | 32×288×512  | 3×288×512   |       99 |
| R-STR   | space-time Transformer residual | 3×288×512   | 3×288×512   |    56288 |

Total parameters: 409,995
