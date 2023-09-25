FILL README

WRITE DESCRIPTION

## Installation

Below are steps of how to install `neuromuscular-control`. We mainly use `poetry` to manage
the project, although most of the important commands will be provided in `Makefile`.

1. Clone!

First **create the fork repository and clone** to your local machine.

2. Virtual python workspace: `conda`, `pyenv`, or `venv`.

We recommend using python version above 3.10.

```bash
conda create --name neuromuscular_control-env
conda activate neuromuscular_control-env
conda install pip
```

3. Setup [`poetry`](https://python-poetry.org) and `dependencies`!

```bash
make poetry-download
make install
```



