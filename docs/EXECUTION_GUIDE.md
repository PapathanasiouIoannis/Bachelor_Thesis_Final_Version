# How to run the Pipeline

This project is split into three main "orchestrators". An orchestrator is just a master script that runs all the smaller scripts in the right order so you don't have to do it manually.

> [!CAUTION]
> **Configuration Rules:** 
> Try not to edit `main.py`, `physics_main.py`, or `perturb_main.py` directly unless you're adding a new script to the sequence. If you want to change hyperparameters, dataset sizes, or file paths, do it inside `src/config.py`.

## 1. The Physics Orchestrator (`physics_main.py`)
This is the first thing you need to run if you don't have any data. It handles all the heavy lifting for the theoretical physics:
- It generates the core Equations of State (EoS) for both Hadronic and Quark matter.
- It solves the Tolman-Oppenheimer-Volkoff (TOV) equations to figure out what the stars would actually look like (Mass, Radius, etc.).
- It saves all this data into fast Parquet files in the `data/` folder.

**Command:**
```bash
python physics_main.py
```

## 2. The Clean ML Orchestrator (`main.py`)
Once you have your theoretical physics data, run this script to train the machine learning models. 
- It loads the Parquet data and trains XGBoost and PyTorch Neural Networks.
- It runs a bunch of advanced evaluations (like UMAP, Confusion Matrices, Calibration).
- It outputs all the weights to `outputs/` and the plots to `plots/`.

**Command:**
```bash
python main.py
```

## 3. The Perturbed ML Orchestrator (`perturb_main.py`)
This is the most realistic part of the thesis. Telescopes aren't perfect, so this script takes the clean physics data and injects realistic Gaussian noise into it (like a 10% error on Radius).
- It retrains all the models specifically to handle this "smudged" data.
- This is where we prove that you really need Gravitational Wave data (Tidal Deformability) to tell Quark stars and Hadronic stars apart when noise is involved!
- It outputs to `outputs_perturb/` and `plots_perturb/`.

**Command:**
```bash
python perturb_main.py
```

## Troubleshooting
If a script crashes, check out `pipeline_debug.log`. It usually has the exact python error and stack trace so you can figure out what went wrong. If a specific ML evaluation script fails, you can always run it individually from the `src/ml/` folder to debug it!
