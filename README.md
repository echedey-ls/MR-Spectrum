Modelling and characterization of usable irradiance
===================================================

To be used with N. Martín and J. M. Ruiz spectral mismatch factor. From Nuria's thesis
3.2b equation:

```math
PS = 1 - \frac{S_{efE(\lambda)}}{S_{ef\bar{G}(\lambda)}}\frac{E_{\lambda<\lambda_0}}{\bar{G}_{\lambda<\lambda_0}}\frac{\bar{G}}{E}
```

where:

 * $`E = \int_{\lambda_{min}}^{+\infty} E(\lambda) d\lambda`$
 * $`E_{\lambda<\lambda_0} = \int_{\lambda_{min}}^{\lambda_0} E(\lambda) d\lambda`$
 * $`\bar{G} = \int_{\lambda_{min}}^{+\infty} G(\lambda) d\lambda`$
 * $`\bar{G}_{\lambda<\lambda_0} = \int_{\lambda_{min}}^{\lambda_0} G(\lambda) d\lambda`$
 * $`G`$ represents the standard (STC) spectrum

In this repo, I will be working on a first modelling of
$`\frac{E_{\lambda<\lambda_0}}{E}`$, which is the ratio of usable spectrum against
all incident spectrum. By *usable* we mean the wavelength until a PV material has
at least some effectiveness at converting irradiance into current.

For example, for `c-Si` it is around $`1100 nm`$ and for `a-Si`, around $`800 nm`$.
See *Figure 3* in [1].

Also, ratio $`\frac{\bar{G}}{\bar{G}_{\lambda<\lambda_0}}`$ is available through
``G_over_G_lambda(cutoff_wavelength)`` in ``Irradiance_ratios/irradiance_ratios.py``.

$`\frac{S_{efE(\lambda)}}{S_{ef\bar{G}(\lambda)}}`$ is already modelled in
[1].


Available workflows
-------------------

1. ``Irradiance_ratios/E_ratio_script.py``: plots $`\frac{E_{\lambda<\lambda_0}}{E}`$
against SPECTRL2 inputs and time-dependant inputs.

References
----------

[1] N. Martín and J. M. Ruiz, ‘A new method for the spectral characterisation of PV modules’,
    Progress in Photovoltaics: Research and Applications, vol. 7, no. 4, pp. 299–310, 1999,
    [doi: 10.1002/(SICI)1099-159X(199907/08)7:4<299::AID-PIP260>3.0.CO;2-0](doi.org/10.1002/(SICI)1099-159X(199907/08)7:4<299::AID-PIP260>3.0.CO;2-0).
