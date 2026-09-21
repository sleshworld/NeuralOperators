# Neural Operators for Computational Fluid Dynamics

Research notebooks for forecasting the evolution of a two-dimensional Taylor-Green vortex with neural operators. The project compares physics-informed DeepONet variants with a Fourier Neural Operator (FNO) on the incompressible Navier-Stokes equations at a high Reynolds number.

This repository accompanies the master's thesis [*Application of Neural Operators to Computational Fluid Dynamics*](https://elib.spbstu.ru/dl/3/2025/vr/vr26-329.pdf/info). A copy of the thesis is included in [`Thesis.pdf`](./Thesis.pdf).


What is implemented

- A pseudo-spectral solver that generates synthetic vorticity trajectories on a periodic square domain.
- A physics-informed DeepONet (PI-DeepONet) that embeds the governing equations in the loss function.
- A transfer-learning variant (PI-TL-DeepONet) for long-horizon rollout.
- An autoregressive Fourier Neural Operator baseline.
- Evaluation and visualization utilities for relative L2 error, rollout stability, inference time, field snapshots, and animations.

The thesis reports the best result for FNO: a relative L2 error of `0.55` on a `32 x 32` grid with an inference time of approximately one second on the evaluated hardware.

## Problem formulation

The notebooks model the two-dimensional incompressible Navier-Stokes equations in vorticity-streamfunction form:

$$
\frac{\partial \omega}{\partial t} + \mathbf{u}\cdot\nabla\omega
= \nu\nabla^2\omega + f,
$$

$$
-\nabla^2\psi = \omega,
\qquad
\mathbf{u} = \left(\frac{\partial\psi}{\partial y}, -\frac{\partial\psi}{\partial x}\right).
$$

Randomized Taylor-Green initial conditions are evolved with a spectral solver. The resulting trajectories provide training and test data for both operator-learning approaches.

## Repository contents

| File | Purpose |
| --- | --- |
| [`Data.ipynb`](./Data.ipynb) | Generate Navier-Stokes trajectories, export DeepONet/FNO datasets, and visualize vorticity and velocity evolution. |
| [`DeepONet.ipynb`](./DeepONet.ipynb) | Define, train, checkpoint, and evaluate PI-DeepONet and PI-TL-DeepONet. |
| [`FNO.ipynb`](./FNO.ipynb) | Train and evaluate the autoregressive 2D FNO model. |
| [`Thesis.pdf`](./Thesis.pdf) | Full methodology, experiments, and discussion. |

## Environment

The notebooks were developed with:

- Python 3.8
- PyTorch 1.7
- TensorFlow 2.4
- NumPy, SciPy, Matplotlib, scikit-learn, h5py, imageio, and tqdm
- NVIDIA GPU support for practical training times

`Data.ipynb` uses the legacy `torch.rfft` and `torch.irfft` APIs. These APIs were removed from later PyTorch releases, so use PyTorch 1.7 or migrate the solver to `torch.fft` before running it in a modern environment.

The FNO notebook currently selects CUDA explicitly. DeepONet training in the recorded experiment took about 16 hours on an NVIDIA V100, so CPU execution is intended only for small checks.

## Running the experiments

1. Clone the repository and start Jupyter from the repository root.
2. Open `Data.ipynb` and review the generation parameters. Set `size = 32` to match the default configurations in the model notebooks, or update every downstream grid setting consistently.
3. Run `Data.ipynb`. It creates a dataset under `Data/<configuration-name>/` in both NumPy and MATLAB formats.
4. Update `npy_name` in `DeepONet.ipynb` and `TRAIN_PATH` / `TEST_PATH` in `FNO.ipynb` to the generated files. The committed notebook paths start with `../Data/`; when Jupyter runs from the repository root, use `Data/` instead.
5. Run either model notebook. Reduce the sample count, rollout length, and epoch count for a smoke test before starting a full training run.

The generated datasets and trained checkpoints are not included in the repository. Reproducing the thesis-scale experiments requires regenerating the data and access to a suitable GPU.

## Citation

If this repository supports your work, cite the accompanying thesis:

> Mikhail S. Troshin. *Application of Neural Operators to Computational Fluid Dynamics*. Master's thesis, Peter the Great St. Petersburg Polytechnic University, 2025. [DOI: 10.18720/SPBPU/3/2025/vr/vr26-329](https://doi.org/10.18720/SPBPU/3/2025/vr/vr26-329).
