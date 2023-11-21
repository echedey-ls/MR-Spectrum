Modelling and characterization of usable irradiance
===================================================

To be used with N. Martín and J. M. Ruiz spectral mismatch factor. From Nuria's thesis
3.2b equation:

```math
PS = 1 - \frac{S_{efE(\lambda)}}{S_{ef\bar{G}(\lambda)}}\frac{E_{\lambda<\lambda_0}}{\bar{G}_{\lambda<\lambda_0}}\frac{\bar{G}}{E}
```

In this repo, I will be working on a first modelling of
$`\frac{E_{\lambda<\lambda_0}}{E}`$, which is a relation of usable spectrum against
all incident spectrum. By *usable* we mean the wavelength until a PV material has
at least some effectiveness at converting irradiance into current.
For example, for `c-Si` it is around $`1100 nm`$ and for `a-Si`, around $`800 nm`$.

Also, calculation of $`\frac{G}{G_{\lambda<\lambda_0}}`$ is available through
``G_over_G_lambda(cutoff_wavelength)`` in ``Irrads_relation/irrads_relation_fracs.py``,
where $`G`$ is the standard (STC) spectrum.

Available workflows
-------------------

1. ``Irrads_relation/E_ratio_script.py``: plots $`\frac{E_{\lambda<\lambda_0}}{E}`$
against SPECTRL2 inputs and time-dependant inputs.
