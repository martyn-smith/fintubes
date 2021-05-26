An exploration and optimisation of microfin tubes
===

Motivation
---

Gain/Solidify experience with modelling and optimisation

Goals
---

Optimise the heat transfer coefficient (h).
Investigate the impacts of a move to low-pressure, HFO, refrigerants.
Compare Cavallini (2009) to a custom model, and find optimised parameters based off these models.

```
Data => Model => Optimiser
```

Dependencies
---

The stack currently uses:

pandas for import and filtering
\+ numpy for correlation matrix
\+ matplotlib and seaborn for plotting
\+ sklearn and keras for model fitting
\+ scipy.optimise was used for optimisation using the Powell method and a maximum 1000 iterations.

Initial Observations
---

The dataset consists of 4333 observations.

![Figure 1: initial correlations](figures/reduced_correlations.png)

*Figure 1: initial correlations*

As we can see, geometric parameters do not correlate strongly to h. Fin apex angle (α) and helix angle (β) correlate negatively with h, fin height (hal) correlates positively with h, and tube length (L) correlates negatively. The operational parameters, mass flow (G) in particular, exert a stronger influence. A simple linear regression with all parameters combined predicts the heat transfer coefficient well, with R<sup>2</sup> of 0.926, and only tube length and latent heat presenting no statistical significance. The dataset is highly collinear, as might be expected, with 16 refrigerants tested and 12 of the parameters highly dependent on refrigerant choice.

![Figure 2: refrigerant performance](figures/refrigerant_violins.png)

*Figure 2: performance by refrigerant and fin type*

Grouped by refrigerant and fin type, R11 (also known as CFC11, used exclusively in this dataset by Nozu et. al) has a clear outlier of high performance, with a maximum of nearly 50 kW.m<sup>-2</sup>K<sup>-1</sup> at a P<sub>reduced</sub> of 0.039. The highest h not using R11 achieves h = 30 kW.m<sup>-2</sup>K<sup>-1</sup>. Low-pressure refrigerant types R11, R113 and R123 are featured in the dataset, and outperform the featured high-pressure refrigerant types R12, R22, R134a, R410a, R502, (h̅ = 8.5 kW.m<sup>-2</sup>K<sup>-1</sup> for low-pressure and 6.5 kW.m<sup>-2</sup>K<sup>-1</sup> for high-pressure respectively, p < 0.01). The vast majority of refrigerants were high-pressure or uncategorised, with n=166 samples of low-pressure refrigerants.

Grouped by fin type alone, crossgrooved significantly outperforms microfin, regardless of the presence of R11 data (h̅ = 6.0 kW.m<sup>-2</sup>K<sup>-1</sup> for microfin and 9.4 kW.m<sup>-2</sup>K<sup>-1</sup> for crossgrooved respectively, p < 0.01). Fin height shows a noticeable peak at 0.00023 m.

Grouping at the 95th percentile (dropping Nozu/CFC11 data), provided an initial seed. One observation, with an unphysical vapour quality of 1.04, was also excluded when applying the Cavallini model.

Fitting (α, β) only
---

An empirical map of (α, β) to h shows  peaks at (44.1 °, 13.1 °), (49.5 °, 20.8 °), (65.0 °, 20.9 °). Several regions, such as α \< 0.28 β - 13.0 ° and α \< -2.3 β + 30.0 °, are unexplored, the latter in particular indicating that axial tubes (β = 0 °) with α \< 30 ° may be optimal, or at least highly performant. Similarly, diameter and fin height are strongly linked (D = 37.1 ± 0.1 hal), with optimal parameters appearing to be in this region. Cavallini (2009) predicts a (44.5 °, 12.0°) peak and (65.0 °, 20.9 °) peak, and the (β = 0 °) slope, but significantly underestimates the maximum possible h, as noted in same paper.

![Figure 3: alpha-beta surface map](figures/alpha_beta_surface_plus_cav.png)

*Figure 3: left: alpha-beta surface map. Right: as predicted by Cavallini (2009)*

A sensitivity analysis for a polynomial fit for (α, β) alone, with no regularisation strength, shows an asymptote of proportion of variance explained of ~0.25 at the third degree. Despite this, higher degrees are needed to avoid divergence during optimisation: a fifth-degree polynomial optimises α and β to (42.1 °, 15.3 °) respectively, and a sixth-degree polynomial to (49.4 °, 10.1 °), respectively, with estimated performances of 7.6 kW.m<sup>-2</sup>K<sup>-1</sup> and 27.4 kW.m<sup>-2</sup>K<sup>-1</sup>. 

| Method | (α, β) |
|--------|------------|
| Empirical | (44.1 °, 13.1 °) |
| Cavallini (2009) | (44.5 °, 12.0°) |
| Polynomial (fith degree) | (42.1 °, 15.3 °) |
| Polynomial (sixth degree) | (49.4 °, 10.1 °) |


![Figure 4: polynomial fit accuracy for geometric parameters, all refrigerants](figures/geometric_poly_rsquared_plus_fifth_surface.png)

*Figure 4: left: polynomial fit accuracy for geometric parameters only, all refrigerants. Right: fith-degree polynomial surface for alpha and beta, mean diameter, fin height and length*

Fitting Expanded Parameters and PCA against Cavallini
---

Available parameters were then expanded to:

Saturation temperature T<sub>sat</sub>
Mass flow rate G
vapour quality x
diameter D
helix angle β
liquid thermal conductivity λ<sub>l</sub>
reduced pressure P<sub>red</sub>

Cavallini's model has the following input-output relation:

Cavallini: 

$\mu_l, \mu_v, \rho_l, \rho_v, C_{p,l}, C_{p,v}, \lambda_l, \Delta H_{fus}, \Delta T, G, x, ng, D, \alpha, \beta, hal \mapsto h$

And our model:

Model: 

$T_{sat}, G, x, D, \beta, \lambda_l, P_{reduced} \mapsto h$

A third-degree polynomial regression (Poly 3) was performed, with a regularisation strength of 1 (although cross-validation indicates lower values may be usable). The train/test split was a 60 \%. A variant compressing the seven input parameters to 6 via PCA before regression (PCA 6 + Poly 3) was also performed.

Finally, two PCA models on all non-categorical parameters were performed, both fitting the same third-degree polynomial kernel ridge regression as before, reducing to either six dimensions (All noncategoricals PCA 6 + Poly 3 ) or five (All noncategoricals PCA 5 + Poly 3 ). 

Results are summarised below. A reconstruction of the alpha-beta-h surface for the poly 3 fit is also below.

| Method | Trainable Parameters | &nbsp; &nbsp; R<sup>2</sup> &nbsp; &nbsp; |
|--------|----------------------|-------------------------------------------|
| Cavallini (2009) | 29 | 0.76 | 
| Cavallini (2009) (CFC11 excluded) | 29 | 0.77 | 
| Poly 3 | 120 | 0.79 |
| Poly 3 (CFC11 excluded) | 120 | 0.80 |
| ANN | 76 | 0.74 |
| ANN (CFC11 excluded) | 76 | 0.70 |


![Figure 6: modelled alpha/beta/h for a third-degree polynomial](figures/alpha_beta_kernel_3.png)

| Method | &nbsp; &nbsp; R<sup>2</sup> &nbsp; &nbsp; | &nbsp; Ts (°C) &nbsp;|&nbsp; G (kg.m<sup>-2</sup>s<sup>-1</sup>) &nbsp;|&nbsp; x &nbsp;|&nbsp; D (m) &nbsp;|&nbsp; β (°) &nbsp;|&nbsp; λ<sub>l</sub> (W.m<sup>-1</sup>K<sup>-1</sup>) &nbsp;|&nbsp; P<sub>reduced</sub> &nbsp;|&nbsp; h<sub>predicted</sub> (kW.m<sup>-2</sup>K<sup>-1</sup>) &nbsp;|
|--------|:------------:|---------|-------------------------------------|---|-------|-------|---------------|---------------------|-----------------------|
| 95th percentile values, no CFC11 | - | 39.0 | 440.0 | 0.85 | 0.0080 | 12.0 | 0.087 | 0.25 |
| 95th percentile values, low-pressure type | - | 40.0 | 230.0 | 0.84 |0.0080 | 20.0 | 0.083 | 0.039 |
| Poly 3 | 0.79 | -15.0 | 919.0 | 1.0 | 0.016 | 0.0 | 0.52 | 0.028 | 46.0 |
| All noncategoricals PCA 6 + Poly 3 | 0.75 | -15.0 | 21.0 | 0.0 | 0.016 | 0.0 | 0.52 | 0.67 | > 100.0 |

Cavallini optimised with TNC method:

| μ<sub>l</sub> (μPa.s) | μ<sub>v</sub> (μPa.s) | ρ<sub>l</sub> (kg.m<sup>-3</sup>) | ρ<sub>v</sub> (kg.m<sup>-3</sup>) | C<sub>p,l</sub> (J.kg<sup>-1</sup>K<sup>-1</sup>) | C<sub>p,v</sub> (J.kg<sup>-1</sup>K<sup>-1</sup>) | λ<sub>l</sub> (W.m<sup>-1</sup>K<sup>-1</sup>) | ΔH<sub>fus</sub> kJ.kg<sup>-1</sup>| ΔT (°C) | &nbsp; G (kg.m<sup>-2</sup>s<sup>-1</sup>) &nbsp;|&nbsp; x &nbsp; | Fin type | &nbsp; D (m) &nbsp; | &nbsp; α (°) &nbsp; | &nbsp; β (°) &nbsp; | &nbsp; hal (m) &nbsp; | &nbsp; h<sub>predicted</sub> (kW.m<sup>-2</sup>K<sup>-1</sup>) |
|-----------|-----------|--------|------|--------|------|------|--------|----|-----|---|---|--------|----|------|-----------|-----|
|110.0 | 13.0 | 1500.0 | 28.0 | 2800.0 | 1400.0 | 0.52 | 220.0 | 31.0 | 770.0 | 1.0 | microfin | 0.0075 | 42.0 | 18.0 | 0.00021 | 70.0|