# Local Labels Beat Public Data: Cross-Network Evaluation of Lightweight and Explainable IoT Intrusion Detectors

Code, notebook, result files and ESP32 benchmark for the paper by **Md Masum Kawsar** and **Mayishat Altab Mridu** (Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh)

> **Local Labels Beat Public Data: Cross-Network Evaluation of Lightweight and Explainable IoT Intrusion Detectors**
> (submitted to *SN Computer Science*)

The study trains five model families and nine microcontroller-sized models on four public NetFlow-v2 datasets that share one feature schema, and evaluates them within datasets, across datasets and leave-one-dataset-out. It then measures how many labelled flows a new network needs, whether public data helps, and how the smallest models perform on an ESP32.

## Main findings

| Finding | Number |
|---|---|
| Within-dataset MCC (RF, XGBoost) | 0.94 |
| Cross-dataset / leave-one-dataset-out MCC (all models) | about 0 (−0.35 to −0.01) |
| Testbed features (TTL, TCP window) in SHAP importance | 14–40% |
| Cross-dataset pairs with ROC-AUC below 0.5 (inverted) | 62% |
| Depth-4 tree, 1000 local labels at the real attack share | MCC 0.79 (best tuned ensemble 0.76) |
| Depth-4 tree on ESP32 | 405 bytes, 2.4 µs per flow, predictions identical to scikit-learn |

## Repository layout

```
notebooks/IoT_IDS_NetFlow_Colab.ipynb   Google Colab notebook that runs everything (Steps 1-8)
src/iot_ids_pipeline.py                 Full experiment pipeline (resumable, checkpointed per seed)
src/prepare_nf_samples.py               Stratified sampling of the full NetFlow-v2 files
paper/make_figs.py                      Draws the paper's figures (Figs. 2–9) from the result CSVs
esp32/esp32_benchmark/                  Arduino sketch: 9 compact models exported to C + 500 test flows
esp32/esp32_results.csv                 Raw ESP32 measurements used in the paper (Table 8)
results/                                All result CSVs of the final run (10 seeds) + run_config.json
results/figures/                        Figures and LaTeX/CSV tables produced by the notebook
results/esp32_verification/             gcc check: C models vs scikit-learn (100% agreement)
```

## Data

The four datasets are **not** included; download them from the University of Queensland:
<https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets>

| Dataset | Network |
|---|---|
| NF-ToN-IoT-v2 | IoT / IIoT |
| NF-BoT-IoT-v2 | IoT botnet |
| NF-UNSW-NB15-v2 | Enterprise |
| NF-CSE-CIC-IDS2018-v2 | Enterprise |

The datasets are free for academic research when the authors' papers are cited (see the UQ page). Please cite:
Sarhan, M., Layeghy, S., Portmann, M.: Towards a standard feature set for network intrusion detection system datasets. *Mobile Networks and Applications* 27, 357–370 (2022). <https://doi.org/10.1007/s11036-021-01843-0>

`prepare_nf_samples.py` draws a fixed-seed stratified sample of about 200,000 flows from each full file (about 50% benign, attack types balanced, rare types kept in full), so the samples used in the paper can be regenerated exactly from the original files.

## How to reproduce

1. Open `notebooks/IoT_IDS_NetFlow_Colab.ipynb` in Google Colab.
2. Put each downloaded dataset in its own folder on Google Drive and set `DATA_V2` in Step 2 to the parent folder.
3. Run Steps 1–3 (install, settings, write scripts), Step 4a (sampling, one time) and Step 4b (column check).
4. Run Step 5 (experiment; resumable after a disconnect), Step 6 (figures and tables) and Step 7 (download).

Settings of the final run: `SEEDS = 10`, `TUNE = False`, `MAX_ROWS = 50000`, all four datasets. A full run takes roughly 10–20 hours of Colab CPU time over several sessions; results are checkpointed per seed and stage.

Software used for the final run (from `results/run_config.json`): Python 3.13, pandas 2.2.3, NumPy 2.4.6, scikit-learn 1.6.1, XGBoost 3.4.1, SHAP.

The pipeline can also be run directly:

```bash
pip install -r requirements.txt
python src/prepare_nf_samples.py --out nf_samples --raw-dir nf_raw \
    --datasets NF-ToN-IoT-v2 NF-BoT-IoT-v2 NF-UNSW-NB15-v2 NF-CSE-CIC-IDS2018-v2
python src/iot_ids_pipeline.py --profile nf --nf-dir nf_samples --max-rows 50000 --seeds 10 --out results_nf
```

Redraw the paper figures from the results: `python paper/make_figs.py results paper_figs`.

## ESP32 benchmark

Hardware: ESP32-WROOM-32 development board (ESP32-D0WD-V3, 240 MHz, 4 MB flash), Arduino core for ESP32.

1. Install the Arduino IDE and the *esp32 by Espressif Systems* board package.
2. Open `esp32/esp32_benchmark/esp32_benchmark.ino`, select **ESP32 Dev Module** and the board's port, and upload.
3. Open the Serial Monitor at 115200 baud and press EN/RST. The sketch prints one `RESULT` line per model (time per prediction, agreement with scikit-learn on 500 flows, free heap) and `DONE`.

The sketch in this repository is the exact one measured in the paper (`esp32/esp32_results.csv`).

## Citation

If you use this code, please cite the paper (details will be added after publication) and the dataset paper above. See `CITATION.cff`.

## Licence

Code: MIT licence (see `LICENSE`). The datasets keep their own terms of use (see the UQ page).
