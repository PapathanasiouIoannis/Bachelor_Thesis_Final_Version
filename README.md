# Neutron Star Equation of State (EoS) Inference Framework

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

## Hi there! Welcome to my Bachelor Thesis project!
This repository contains the full code for my thesis project on Neutron Star Equation of State (EoS) Inference. 

The goal of this project is to figure out what happens inside the extremely dense core of a neutron star. Specifically, we want to know if the core is made of standard "Hadronic" matter (protons, neutrons, etc.) or if it breaks down into exotic "Quark" matter. Since we can't look inside a neutron star, we use a combination of theoretical physics and machine learning to make predictions based on what telescopes *can* see from the outside.

## How it works
The project operates in a few major steps:
1. **Physics Generation**: We simulate thousands of possible neutron stars by solving the Tolman-Oppenheimer-Volkoff (TOV) equations. We use different theoretical models (Hadronic vs. Quark) to see what kinds of masses, radii, and tidal deformabilities they would produce.
2. **Clean Machine Learning**: We train machine learning models (XGBoost and PyTorch MLPs) on this perfect theoretical data to see if we can classify the core state just by looking at Mass ($M$), Radius ($R$), and Tidal Deformability ($\Lambda$).
3. **The "Masquerade" Problem (Perturbed ML)**: Real telescopes like NICER and LIGO have massive error margins (noise). We inject realistic Gaussian noise into our physics data to simulate real-world observations and retrain our models. This proves that adding Gravitational Wave data ($\Lambda$) is crucial for breaking the "masquerade" where Quark stars look identical to Hadronic stars.

## Setup
It's highly recommended to use a virtual environment so you don't mess up your system Python packages.

```bash
# 1. Create the virtual environment
python -m venv .venv

# 2. Activate it
# On Windows:
.venv\Scripts\activate
# On MacOS/Linux:
source .venv/bin/activate

# 3. Install the packages
pip install -r requirements.txt
```

## How to run the code
The pipeline is split into three main "orchestrators" so you don't have to run everything manually.

### 1. Physics Generation
Run this to generate the raw Equations of State and solve the TOV equations. It outputs the data into Parquet files.
```bash
python physics_main.py
```

### 2. Clean Machine Learning Pipeline
This trains the XGBoost and Neural Network models on the clean, theoretical data. It also generates all the performance plots and visualizations.
```bash
python main.py
```

### 3. Perturbed (Noisy) Machine Learning Pipeline
This injects realistic telescope noise into the data and trains specialized models to handle the uncertainty.
```bash
python perturb_main.py
```

## Interactive Dashboards
I built two interactive Streamlit dashboards so you can play around with the models and see how they predict the core state based on custom telescope measurements.

To run the Clean Dashboard:
```bash
streamlit run app_ref.py
```

To run the Noisy/Perturbed Dashboard (which is much more realistic):
```bash
streamlit run perturb_app_ref.py
```

## Note on "Compactness"
In earlier iterations of this project, we used Compactness ($C = M/R$) as a feature. We eventually realized this causes massive data leakage and gives the models an unfair advantage. We have completely stripped $C$ out of the final pipelines to ensure our results are fully grounded in realistic macroscopic observables!
