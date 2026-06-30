# Execution Guide for the Pipeline

The thesis code is split into three main orchestrator files. You have to run them in a specific order so that the datasets are generated before the machine learning tries to train on them. I structured it this way to keep the physics generation separate from the classification tasks.

### Step 1: Data Synthesis
First you run the physics engine.
```bash
python physics_main.py
```
This script solves the TOV equations and generates the theoretical equations of state for both hadronic and quark stars. It saves a large file called `physics_test_dataset.parquet` in the `data/` directory. Be aware it might take a while if you set the sample size very high in the config.

### Step 2: Clean Machine Learning Training
Next, run the main classification pipeline.
```bash
python main.py
```
This loads the parquet dataset and trains the XGBoost and BNN models. It evaluates the classification performance on the clean theoretical data. All the model weights will dump into `models/` and the evaluation plots (like ROC and SHAP diagrams) will save into `plots/`.

### Step 3: Observational Noise Simulation
Finally, run the perturbation pipeline.
```bash
python perturb_main.py
```
This step is critical for the masquerade problem analysis. It takes the clean dataset and injects gaussian noise into the mass and radius variables, simulating what a real telescope actually measures. Then it retrains the models on this noisy data to prove that the tidal deformability parameter is required to maintain accuracy. The output plots for this go into `plots_perturb/`.
