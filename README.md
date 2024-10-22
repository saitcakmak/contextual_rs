# Contextual Ranking and Selection with Gaussian Processes

[S. Cakmak, Y. Wang, S. Gao, and E. Zhou. Contextual Ranking and Selection with Gaussian Processes and OCBA. ACM Transactions on Modeling and Computer Simulation, 2023.](https://dl.acm.org/doi/pdf/10.1145/3633456)

```
@article{cakmak2023contextual,
  title={Contextual Ranking and Selection with Gaussian Processes and OCBA},
  author={Cakmak, Sait and Wang, Yuhao and Gao, Siyang and Zhou, Enlu},
  journal = {ACM Transactions on Modeling and Computer Simulation},
  year={2023},
  publisher={Association for Computing Machinery},
  address = {New York, NY, USA},
}
```

[S. Cakmak, S. Gao, and E. Zhou. Contextual Ranking and Selection with Gaussian 
Processes. Winter Simulation Conference, 2021.](https://arxiv.org/abs/2007.05554)

```
@inproceedings{cakmak2021contextual,
  title={Contextual Ranking and Selection with Gaussian Processes},
  author={Cakmak, Sait and Gao, Siyang and Zhou, Enlu},
  booktitle = {Proceedings of the 2021 Winter Simulation Conference},
  year={2021},
}
```


### Installation:
It is recommended to install `PyTorch` via Conda. See [PyTorch installation 
instructions](https://pytorch.org/get-started/locally/).

```bash
conda install pytorch -c pytorch
pip install -e .
```

### Repo Structure:
* The `WSC` brach includes the code for the WSC version of the paper.
* The `TOMACS` branch includes additional algorithms and experiments, which are 
  presented in the journal article.

### Directory:
Any folder that is not noted below is safe to ignore.

- The `contextual_rs` folder defines the algorithms and models used in the paper / 
  experiments, as well as some experimental ones that didn't make it to the paper. The 
  main logic of the GP-C-OCBA is defined in `contextual_rs_strategies` as the 
  `gao_modellist` function. Provided with a ModelListGP, this returns the next 
  arm-context to sample from. Refer to the experiments for actual usage. The LEVI
  algorithm is defined in `levi.py`, and the continuous extension of GP-C-OCBA is
  implemented in `continuous_context.py`.
- The `experiments` folder contains the main scripts for running the experiments and 
  the corresponding experiment output. The experiments presented in the WSC paper are 
  found in the `wsc_experiments` folder and were run using the `main.py` (and other 
  helpers). The docstrings and in-line comments in `main.py` were updated to explain 
  what does what. It should be relatively straightforward to follow, though I would be 
  happy to help out where needed. Feel free to open up a GitHub issue with any questions!


NOTE: The runtimes reported for LEVI in the paper come from the experiment outputs labeled
`<seed>_LEVI.pt` and the actual results come from the outputs labeled `<seed>_LEVI_new.pt`.
This is due to some changes that were made to model fitting logic before running LEVI_new,
which significantly reduced the common overhead shared with the other method. To keep things
comparable, we reported the runtimes obtained using the previous version.
