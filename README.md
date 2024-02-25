N. Martin & J. M. Ruiz Spectral Mismatch Factor
===============================================

In this repo I am developing a Spectral Mismatch Factor for `a-Si`, `c-Si` and `m-Si` PV
modules, following a procedure developed by Nuria Martín & José María Ruiz.

This analysis is inspired and mimics most of the workflow of Nuria's PhD thesis, but the
spectral data is get from spectrum simulators.

Modelling and characterization of usable irradiance
---------------------------------------------------

From Nuria's thesis 3.2b equation:

```math
    PS = 1 - \frac{S_{efE(\lambda)}}{S_{ef\bar{G}(\lambda)}}\frac{E_{\lambda<\lambda_0}}{\bar{G}_{\lambda<\lambda_0}}\frac{\bar{G}}{E}
```

where:

 * $`E = \int_{\lambda_{min}}^{+\infty} E(\lambda) d\lambda`$
 * $`E_{\lambda<\lambda_0} = \int_{\lambda_{min}}^{\lambda_0} E(\lambda) d\lambda`$
 * $`\bar{G} = \int_{\lambda_{min}}^{+\infty} G(\lambda) d\lambda`$
 * $`\bar{G}_{\lambda<\lambda_0} = \int_{\lambda_{min}}^{\lambda_0} G(\lambda) d\lambda`$
 * $`G`$ represents the standard (STC) spectrum

$`\lambda_0`$ is what I call the _cutoff wavelength_, the wavelength until a PV material has
at least some effectiveness at converting irradiance into current.
For example, for `c-Si` and `m-Si` it is around $`1100 nm`$ and for `a-Si`, around $`800 nm`$.
See *Figure 3* in [[1]](#references).

This means the ratio $`\frac{\bar{G}}{\bar{G}_{\lambda<\lambda_0}}`$ is constant for any
$`\lambda_0`$. It is available through ``G_over_G_lambda(cutoff_wavelength)`` in
``research/irradiances_ratios/ratios_calculator.py``.

In this repo, I will be working on a first modelling of
$`\frac{E_{\lambda<\lambda_0}}{E}`$, which is the ratio of usable spectrum against
all incident spectrum. By *usable* we mean the wavelength until a PV material has
at least some effectiveness at converting irradiance into current.

$`\frac{S_{efE(\lambda)}}{S_{ef\bar{G}(\lambda)}}`$ is already modelled in
[[1]](#references).

How to use and develop this repo
--------------------------------

1. Run `pip -m install -e .[dev]` to install the common dependency for the `scripts/` folder.
    It's a package that groups almost all the important _backend_ computations.
    The editable switch and the `[dev]` dependency are optional of course;
    remove them if you don't plan on doing any changes inside the `research/` folder.
2. Run `pre-commit install` to add the `pre-commit` hooks.
    The most important feature is formatting and linting with `ruff` on each commit.
3. You are now set up to run any of the workflows in the `scripts/` folder.

Available workflows
-------------------

1. ``scripts/E_ratio_script.py``: main model development happens here
    * Plots $`\frac{E_{\lambda<\lambda_0}}{E}`$ against SPECTRL2 inputs and other outputs.
    * Plots each usable part vs. full integral of each component.
2. ``scripts/plot_martin_ruiz_mismatch.py``: allows testing of the developed model against
    SAPM and First Solar models. Find these functions at ``pvlib.spectrum.mismatch``.

References
----------

[1] N. Martín and J. M. Ruiz, ‘A new method for the spectral characterisation of PV modules’,
    Progress in Photovoltaics: Research and Applications, vol. 7, no. 4, pp. 299–310, 1999,
    [doi: 10.1002/(SICI)1099-159X(199907/08)7:4<299::AID-PIP260>3.0.CO;2-0](https://doi.org/10.1002/(SICI)1099-159X(199907/08)7:4<299::AID-PIP260>3.0.CO;2-0).
