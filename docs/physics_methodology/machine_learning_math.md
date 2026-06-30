# The Machine Learning Pipeline (src/ml/ and src/ml_perturb/)

This is the core of the thesis! I wrote two orchestrators (main.py and perturb_main.py) that handle the machine learning.

### The Models
I chose to use **XGBoost** and **PyTorch MLPs (Neural Networks)**. XGBoost is incredible at handling tabular data and finding complex non-linear splits, while the MLP serves as a deep learning baseline. 

### The Clean vs Perturbed Approach
In the clean pipeline (main.py), the models are trained on perfect theoretical data. They learn the exact boundary between Hadronic and Quark stars.
However, real astronomy is messy. In perturb_main.py, I wrote a data_pipeline.py script that injects Gaussian noise into the measurements (simulating a 10% error on telescope Radius measurements, for example). 

This is where the magic happens: I proved that if you only use Mass and Radius, the noise completely destroys the model's accuracy (because the Quark and Hadronic clouds overlap—the Masquerade problem!). But when you add $\Lambda$ (from Gravitational Waves), the model can disentangle the clouds and maintain high accuracy.
