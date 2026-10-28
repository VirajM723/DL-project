# Tools and environment

| item                          | value                                                             |
|:------------------------------|:------------------------------------------------------------------|
| Platform                      | Kaggle Notebook                                                   |
| Hardware                      | 2 × Tesla T4 | CPU threads 4                                      |
| Python                        | 3.13.15                                                           |
| PyTorch                       | 2.11.0+cu128 (CUDA 12.8, cuDNN 91900)                             |
| NumPy / pandas                | 2.1.3 / 2.3.3                                                     |
| OpenCV / Matplotlib           | 4.14.0 / 3.10.0                                                   |
| FLOP counting                 | torch.utils.flop_counter.FlopCounterMode                          |
| Profile                       | gpu — 512×288, width 0.5, 3000 steps (+1500 fine-tune) × batch 16 |
| Trained models re-used        | 6 (from /kaggle/input)                                            |
| Synthetic training (6 models) | 216.4 min                                                         |
| Real fine-tuning (6 models)   | 111.1 min                                                         |
| Whole notebook                | 127.5 min                                                         |

Libraries: PyTorch, NumPy, pandas, OpenCV, Matplotlib, imageio, Jupyter; run on Kaggle Notebooks (GPU) and tested locally with pytest.
