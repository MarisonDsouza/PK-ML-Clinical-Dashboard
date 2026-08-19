# Mathematical & Architectural Specifications

## 1. The Grey-Box Pharmacokinetic (PK) Paradigm

In classical pharmacokinetics, the distribution of remifentanil is described by a 3-compartment mammillary differential equation system:

$$\frac{dA_1}{dt} = R(t) - (k_{10} + k_{12} + k_{13})A_1 + k_{21}A_2 + k_{31}A_3$$
$$\frac{dA_2}{dt} = k_{12}A_1 - k_{21}A_2$$
$$\frac{dA_3}{dt} = k_{13}A_1 - k_{31}A_3$$
$$\frac{dC_e}{dt} = k_{e0}(C_p - C_e)$$

where:
- $A_1, A_2, A_3$ are amounts in central, shallow peripheral, and deep peripheral compartments ($\mu\text{g}$).
- $C_p = \frac{A_1}{V_1}$ is plasma concentration ($\text{ng/mL}$).
- $C_e$ is effect-site concentration ($\text{ng/mL}$).
- $R(t)$ is infusion rate ($\mu\text{g/min}$).

### Minto Covariate Equations (Minto et al., 1997)
- Volumes: $V_1(Age, LBM)$, $V_2(Age, LBM)$, $V_3(Age, LBM)$
- Clearances: $Cl_1(Age, LBM)$, $Cl_2(Age, LBM)$, $Cl_3(Age, LBM)$
- Rate constants: $k_{ij} = \frac{Cl_{ij}}{V_i}$, $k_{e0}(Age)$
- Lean Body Mass ($LBM$) computed via the James (1976) formula:
  $$LBM_{male} = 1.1 \cdot W - 128 \left(\frac{W}{H}\right)^2$$
  $$LBM_{female} = 1.07 \cdot W - 148 \left(\frac{W}{H}\right)^2$$

---

## 2. Machine Learning Residual Formulation

Instead of training a neural network or tree ensemble directly on observed concentrations $DV$ (which can output negative values or violate mass conservation), the learner predicts a multiplicative log-residual:

$$r = \ln(DV) - \ln(C_{p,\text{minto}})$$

The final inferred concentration is guaranteed non-negative:

$$\hat{C}_p(t) = C_{p,\text{minto}}(t) \cdot \exp(\hat{r}(t))$$

### Feature Vector $\mathbf{x}(t)$
Learners receive 17 dynamic covariates:
1. Mechanistic state features: $C_{p,\text{minto}}(t), C_{e,\text{minto}}(t), C_{2,\text{minto}}(t), C_{3,\text{minto}}(t), \ln(C_{p,\text{minto}}(t))$
2. Patient demographics: $Age, Weight, Height, Sex, BMI, LBM, BSA$
3. Dosing history: $Rate, Rate_{/kg}, Cum\_Dose, is\_infusion, time\_post\_infusion$

---

## 3. Split-Conformal Uncertainty Bounds

For patient safety in clinical TCI, point predictions are augmented with **distribution-free 95% split-conformal prediction intervals**:

$$\hat{C}_p^{low}(t) = \hat{C}_p(t) \cdot \exp(-q_{1-\alpha})$$
$$\hat{C}_p^{high}(t) = \hat{C}_p(t) \cdot \exp(q_{1-\alpha})$$

where $q_{1-\alpha}$ is the $(1-\alpha)$-quantile of absolute log-residuals $|r_i - \hat{r}_i|$ on a held-out calibration set.
