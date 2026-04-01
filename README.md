# DRUM (Diffusion-based Runoff Model)

Code and models for the paper
**“Probabilistic Diffusion Models Advance Extreme Flood Forecasting”** (*Geophysical Research Letters*, [DOI: 10.1029/2025GL115705](https://doi.org/10.1029/2025GL115705)).

---

## Contents

* **codes/** – Model architecture, training, and sampling
* **data_example/** – Example data
* **final_models/** – Pre-trained diffusion models
* **plot_data/** – Data and scripts for figure generation

* `plot_figure2.ipynb`: Reproduces Figure 2 in the GRL paper
* `results_nowcasting/`, `CRPS/`, `shp/`: Supporting data and metrics
* `basin_list.txt`: Basin IDs used in analysis
* `train_conditional_nowcast.ipynb`: Training notebooks for conditional diffusion models
* `train_unconditional_nowcast.ipynb`: Training notebooks for unconditional diffusion models

---

## Usage

* Run `codes/DDIM.ipynb` for DDIM sampling examples
* Run `train_conditional_nowcast.ipynb` / `train_unconditional_nowcast.ipynb` to train diffusion models
* Run `plot_data/plot_figure2.ipynb` to reproduce **Figure 2**

---

## Citation

> Ou, Z., Nai, C., Pan, B., Zheng, Y., Shen, C., Jiang, P., ... & Pan, M. (2025).
> *Probabilistic diffusion models advance extreme flood forecasting.*
> *Geophysical Research Letters, 52*(15), e2025GL115705.
> [https://doi.org/10.1029/2025GL115705](https://doi.org/10.1029/2025GL115705)

