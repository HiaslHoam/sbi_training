# FASTER training set: JWST NIRSpec G395H transmission spectra (cloud-free)

The simulation grid used to train the neural ratio estimation (SBI) networks in our
paper comparing SBI with nested sampling for six hot Jupiters (WASP-94Ab, TrES-4b,
WASP-15b, HAT-P-30b, NGTS-2b, KELT-7b).

- 100,000 noise-free model transmission spectra.
- 61 wavelength bins.
- 12 parameters per spectrum.
- Split exactly as in our training: 90,000 training and 10,000 validation spectra.

Contact: Matthias Heim (LMU Munich), heim.ma@lmu.de.

## Files

| File                     | Contents                                                                                                                          |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `train.npz`              | 90,000 spectra and their parameters (used for training)                                                                           |
| `validation.npz`         | 10,000 spectra and their parameters. Used to pick the best epoch and as the test set for the true-vs-inferred plots in the paper. |
| `noise_baseline_ppm.txt` | Baseline noise curve σ_base(λ) used in the paper: wavelength [µm], σ [ppm]                                                        |
| `noise_model_A_ppm.txt`  | Alternative noise curve (model A): the mean of the six planets' normalised reported errors, scaled so that WASP-94Ab has ω = 1    |
| `omega_per_planet.json`  | Noise scaling ω for each planet: the fitted values for model A and the values used in the paper with the baseline curve           |

## Loading

The files need only NumPy. Each `.npz` file holds one array per key:

```python
import numpy as np
data = np.load('train.npz')
spectra = data['transdep']                  # (90000, 61), ppm
wavelength = data['wavelength_um']          # (61,), µm
theta = np.stack([data[k] for k in ['lg_H2O', 'lg_CO2', 'lg_CO', 'lg_SO2', 'lg_H2S',
                                    'lg_CH4', 'lg_HCN', 'T0', 'offset']], axis=1)   # (90000, 9)
```

## Contents and dimensionality

All arrays are float32. N = 90,000 in `train.npz` and 10,000 in `validation.npz`.

| Key                                                                 | Shape   | Unit                      | Range (uniform prior) | Meaning                                                                                               |
| ------------------------------------------------------------------- | ------- | ------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------- |
| `transdep`                                                          | (N, 61) | ppm                       | about 900 to 40,000   | Transit depth (R*p/R*\*)², noise-free, offset included (see below)                                    |
| `wavelength_um`                                                     | (61,)   | µm                        | 2.78 to 5.18          | Bin centres. Bins 0–29 are NRS1, bins 30–60 are NRS2 (the gap at about 3.7–3.8 µm lies between them). |
| `lg_H2O`, `lg_CO2`, `lg_CO`, `lg_SO2`, `lg_H2S`, `lg_CH4`, `lg_HCN` | (N,)    | log10 volume mixing ratio | −12 to −2             | Abundances, constant with altitude                                                                    |
| `T0`                                                                | (N,)    | K                         | 500 to 2000           | Isothermal temperature                                                                                |
| `offset`                                                            | (N,)    | ppm                       | −150 to +150          | Offset between the detectors, added to the NRS1 bins (0–29)                                           |
| `rp`                                                                | (N,)    | R_Jup                     | 0.6 to 2.0            | Planet radius                                                                                         |
| `rs`                                                                | (N,)    | R_Sun                     | 1.2 to 2.0            | Stellar radius                                                                                        |
| `logg`                                                              | (N,)    | log10(g / cm s⁻²)         | 2.5 to 3.3            | Surface gravity of the planet                                                                         |

**Priors.** All parameters were drawn independently and uniformly over the ranges
above (uniform in log10 for the abundances). Physically unlikely combinations of
`rp`, `rs` and `logg` are therefore included.

**Parameters we infer.** The network infers 10 parameters:

- the 7 abundances, `T0` and `offset`, all in this file;
- `lg_varsig`, the noise inflation, which is not in the files: it is drawn during training (next section).

`logg` and `rs` are known for each real planet and are given to the network as
additional (conditioning) inputs. `rp` is a nuisance parameter that is marginalised over.

## Forward model

The spectra were computed with BeAR in transmission. Settings:

- **Atmosphere:** 100 pressure levels, from 10 bar to 10⁻⁶ bar. Isothermal, cloud-free, constant gravity. Background gas H₂/He; the mean molecular weight follows from the composition.
- **Opacities:**
  - H₂O: POKAZATEL
  - CO₂
  - CO: Li et al. 2015
  - SO₂: ExoAmes
  - CH₄: YT34to10
  - HCN: Harris
  - H₂S: AYT2
  - collision-induced absorption of H₂–H₂ and H₂–He
- **Spectral resolution:** computed at constant R = 10,000, then binned to the 61 R ≈ 100 bins of the NIRSpec G395H data reduction.

## Noise (added during training, not included in the files)

The spectra are noise-free. During training we add new Gaussian noise every epoch,
10 noise draws per spectrum per epoch:

```
obs = transdep + N(0, σ²),    σ(λ) = sqrt( (ω · σ_base(λ))² + ς² )
ω ~ Uniform(0.5, 3)            scales the noise curve σ_base (noise_baseline_ppm.txt or noise_model_A_ppm.txt)
log10 ς ~ Uniform(−1, log10 200)   extra white noise in ppm; lg_varsig = log10 ς is inferred
```

`ω` is also given to the network as an input. For a real planet we use its value
from `omega_per_planet.json`.

```python
rng = np.random.default_rng(0)
sigma_base = np.loadtxt('noise_baseline_ppm.txt')[:, 1]
omega = rng.uniform(0.5, 3, size=(len(spectra), 1))
lg_varsig = rng.uniform(-1, np.log10(200), size=(len(spectra), 1))
sigma = np.sqrt((omega * sigma_base) ** 2 + (10 ** lg_varsig) ** 2)
obs = spectra + rng.normal(size=spectra.shape) * sigma
```

## Good to know

- **Offset convention.** `offset` is already included in `transdep`, on bins 0–29 (NRS1). Bins 30–60 (NRS2) are unchanged. A model of the offset-free spectrum must subtract `offset` from bins 0–29.
- **Validation set as test set.** The validation set also served as the test set for the paper's true-vs-inferred plots. If you choose checkpoints by their validation loss, hold out part of `train.npz` for an independent test.
- **Real observations.** The real spectra of the six planets are binned to the same 61 wavelengths.
