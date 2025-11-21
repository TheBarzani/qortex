# Classical Generator–Discriminator (GAN) Pipeline

This folder contains the **classical counterpart** of the quantum generator–discriminator architecture our teammates are building.

The goal is to have:

- A **conditional generator** that produces next-day swaption surfaces given a context.
- A **discriminator** that learns to distinguish real vs generated surfaces, given the same context.
- **Explicit control over the number of trainable parameters**, so we can match the capacity of the quantum models and compare fairly.

---

## Architecture Overview

We model a **conditional GAN**:

- **Context**: current-day swaption surface (flattened vector of 224 features).
- **Target**: next-day swaption surface (same dimension).

### Generator `G(context, noise) -> next_surface`

- Input:
  - `context`: current surface (dimension = `context_dim`)
  - `noise`: latent vector (dimension = `latent_dim`, e.g. 16)
- Output:
  - `next_surface`: generated next-day surface (dimension = `target_dim`, same as context)

Implemented as an MLP:
- Concatenated input: `[context, noise]`
- Hidden layers: configurable (default `(128, 128)`)
- Output layer: linear mapping to `target_dim`

### Discriminator `D(context, surface) -> logit`

- Input:
  - `context`: current surface (dimension = `context_dim`)
  - `surface`: candidate next-day surface (real or generated)
- Output:
  - Scalar logit (before sigmoid) for “real vs fake”

Implemented as an MLP:
- Concatenated input: `[context, surface]`
- Hidden layers: configurable (default `(128, 128)`)
- Output: single logit

---

## Files

- `models.py`
  - `Generator` and `Discriminator` PyTorch modules.
  - Both are fully parameterized MLPs with easily countable parameters.

- `dataset.py`
  - `SwaptionGANDataset`: wraps `(context, target)` arrays for PyTorch.

- `utils.py`
  - `count_parameters(model)`: exact number of trainable parameters.
  - `set_seed(seed)`: reproducibility for numpy, random, and torch.
  - `get_device()`: selects CUDA if available, else CPU.

- `train_gan.py`
  - Full training loop for the conditional GAN:
    - Loads data via `data_loading.py` (same as other pipelines).
    - Uses current surface as context, next-day surface as target.
    - Scales surfaces using global mean/std.
    - Trains:
      - Discriminator to distinguish real vs fake surfaces.
      - Generator to fool the discriminator.
    - Prints generator and discriminator parameter counts.
    - Saves the trained generator to:
      - `data/Track2_QML/generator_classical.pt`

- `sample_gan_14day.py`
  - Uses the trained generator to **roll out a 14-day trajectory**:
    - Start from the last real surface in `train.xlsx`.
    - Iteratively generate day `t+1` conditioned on day `t`.
    - Saves a CSV with dates and 224-dimensional surfaces:
      - `data/Track2_QML/gan_14day_samples.csv`.

---

## Parameter Matching with Quantum Models

Use:

```python
from generator_discriminator.models import Generator, Discriminator
from generator_discriminator.utils import count_parameters

gen = Generator(context_dim=..., target_dim=..., latent_dim=..., hidden_sizes=(...))
disc = Discriminator(context_dim=..., target_dim=..., hidden_sizes=(...))

print("Generator params:", count_parameters(gen))
print("Discriminator params:", count_parameters(disc))