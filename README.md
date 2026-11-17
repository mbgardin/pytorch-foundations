# PyTorch foundations

Learn PyTorch by reading, running, and changing small examples. The materials assume basic Python, NumPy, introductory machine learning, and familiarity with scikit-learn pipelines. No GPU, account, data download, or prior PyTorch experience is needed after installation.

## Get started

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and download or clone this repository.
2. In a terminal, change to the repository directory and run:

   ```bash
   uv sync
   uv run jupyter lab
   ```

3. Open a notebook in `notebooks/`, select the project's `.venv` Python kernel if prompted, and run its cells from top to bottom. The notebooks find the repository whether Jupyter starts in the repository root or the `notebooks` directory. Restart the kernel before rerunning a notebook from the beginning.

The project uses Python 3.11 and CPU PyTorch by default. Installing packages requires internet access and can take several minutes. The notebooks and data run offline after setup. See [setup and troubleshooting](guides/01-setup.md) for VS Code, device choices, and common errors.

## Learning path

Each notebook is self-contained: it does not depend on variables or saved output from another notebook. Follow the suggested order for the first pass. Exercises invite you to modify and extend the examples.

| Notebook | What you will learn |
| --- | --- |
| [01 Tensors](notebooks/01_tensors.ipynb) | Shapes, indexing, broadcasting, matrix products, reshaping, dtype, devices, and NumPy |
| [02 Automatic differentiation](notebooks/02_autograd.ipynb) | Computation graphs, chain rule, gradients, accumulation, and manual updates |
| [03 Data and batches](notebooks/03_data_and_batches.ipynb) | CSV data, train-only preprocessing, Dataset, DataLoader, shuffling, and batch shapes |
| [04 Models and losses](notebooks/04_models_and_losses.ipynb) | Layers, `nn.Module`, logits, losses, parameters, and classification versus regression |
| [05 Training and evaluation](notebooks/05_training_and_evaluation.ipynb) | Explicit optimizer steps, epochs, validation, metrics, and learning curves |
| [06 Regularization](notebooks/06_regularization.ipynb) | Dropout, weight decay, training-only augmentation, evaluation mode, and early stopping |
| [07 Saving and inference](notebooks/07_saving_and_inference.ipynb) | Save and reload weights, preprocessing, architecture configuration, and predictions |
| [08 Your own model](notebooks/08_your_own_model.ipynb) | An end-to-end baseline and a scaffold for trying architectures and your own data |

The [concept guide](guides/02-concepts.md) connects the notebooks to familiar scikit-learn ideas. The [custom data guide](guides/03-custom-data.md) explains the expected CSV format and common changes for other tasks. The deliberately fictional [dataset](data/README.md) is included locally, with its generation script.

## A complete training script

After working through the notebooks, run `uv run python scripts/train_model.py` for a complete classification example. Edit the settings near the top of [the script](scripts/train_model.py) to try different features, model widths, dropout, batch sizes, or learning rates; modify `Classifier` to explore architectures. The script fits preprocessing on training rows, selects an epoch using validation loss, saves a checkpoint and learning history, and reloads the checkpoint for an example prediction. Each run replaces `artifacts/training_example.pt` and `artifacts/training_history.csv`.

Once you have chosen your settings using validation, run `uv run python scripts/train_model.py --test` for a held-out test assessment. Avoid using test results to choose settings. For a new CSV, update the path, feature and label settings, and split column; the script expects numeric features, integer class indices, and preassigned `train`, `validation`, and `test` rows. See the [custom data guide](guides/03-custom-data.md) for adapting those assumptions and interpreting results.

## A useful mental model

An sklearn pipeline often hides preprocessing and optimization behind `fit`. In PyTorch, you can see and control the steps: read data, fit preprocessing on training data, make batches, define a model and loss, compute gradients, update parameters, and evaluate on held-out data. More control also means more responsibility for avoiding leakage and selecting the correct model mode.

## Practice and sharing

Work through the exercises at the end of each notebook. For a larger project, use notebook 08 as a starting point and describe your data split, preprocessing, model choices, and held-out results. `artifacts/` and the environment are ignored by Git; do not commit private data or checkpoints you do not intend to share.

To check the supplied notebooks in fresh kernels without saving their outputs, run `uv run python scripts/check_notebooks.py` from the repository root. The saving example creates an ignored checkpoint in `artifacts/`.

## References

- [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch: Datasets and DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch: Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [uv: Working on projects](https://docs.astral.sh/uv/guides/projects/)
- [uv: Using PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
