We provide a theoretical interpretation explaining why ESAug improves generalization. Analysis using the sketch loss (analogous for extrusion loss):

1. **Preliminaries.**

   The PLM defines a family of extrusion-induced transformations:

   $$
   \mathcal{T} = \{ T_E : E \in \mathcal{E}_{valid} \},
   $$

   where $T_E(\cdot)$ maps a 2D sketch into 3D space under extrusion $E$, and $\mathcal{E}_{valid}$ denotes the set of valid extrusion sequences.

2. **Ambiguity under ground-truth extrusion.**

   Let $S$ be the ground-truth sketch, $\hat{S}$ the predicted sketch. Sketch loss without ESAug:

   $$
   L_{E_{gt}}(\hat{S}) = \| T_{E_{gt}}(\hat{S}) - T_{E_{gt}}(S) \|^2
   $$

   There may exist $\hat{S} \neq S$ such that:

   $$
   T_{E_{gt}}(\hat{S}) = T_{E_{gt}}(S)
   $$

   resulting in spurious solutions aligned only in a specific coordinate system.

3. **ESAug removes spurious solutions.**

   With ESAug, the loss becomes:

   $$
   L_{\text{ESAug}}(\hat{S}) = \mathbb{E}_{E \sim p_{\text{aug}}} \big[ \| T_E(\hat{S}) - T_E(S) \|^2 \big],
   $$

   where $p_{\text{aug}}$ denotes the augmentation-induced distribution over valid extrusion parameters. If $\mathcal{T} = \{T_E\}$ is sufficiently rich, then for any $\hat{S} \neq S$, there exists at least one $E \in \mathcal{E}_{valid}$ such that:

   $$
   T_E(\hat{S}) \neq T_E(S).
   $$

   In practice, when $p_{\text{aug}}$ sufficiently covers $\mathcal{E}_{valid}$, this implies:
   $$
   L_{\text{ESAug}}(\hat{S}) > 0 \quad \forall \hat{S} \neq S \;\; (\text{approximately}),
   $$
    which is a standard approximation when Monte Carlo sampling is used to estimate expectations over $\mathcal{T}$. ESAug suppresses spurious minima and encourages the solution $\hat{S}$ to be consistent across a broad set of extrusion-induced transformations. This effectively constrains the hypothesis space to transformation-consistent solutions.

4. **ESAug improves generalization:**

   In this analysis, we employ the Empirical Risk Minimization (ERM) framework [1]. At test time, the extrusion sequences may differ from those seen during training. The test risk can be written as:

   $$
   L_{\text{test}}(\hat{S}) = \mathbb{E}_{E \sim p_{\text{test}}} \big[ \| T_E(\hat{S}) - T_E(S) \|^2 \big]
   $$

   By adding and subtracting the training expectation, we obtain:

   $$
   L_{\text{test}}(\hat{S}) = L_{\text{ESAug}}(\hat{S}) + \Big( \mathbb{E}_{E \sim p_{\text{test}}}[\ell(\hat{S},E)] - \mathbb{E}_{E' \sim p_{\text{aug}}}[\ell(\hat{S},E')] \Big),
   $$

   where $\ell(\hat{S},E) = \| T_E(\hat{S}) - T_E(S) \|^2$. Supposing $p_{\text{test}}=\mathcal{E}_{valid}$, this leads to the bound:

   $$
   L_{\text{test}}(\hat{S}) \le L_{\text{ESAug}}(\hat{S}) + \sup \left| \mathbb{E}_{E \sim \mathcal{E}_{valid}}[\ell(\hat{S},E)] - \mathbb{E}_{E' \sim p_{\text{aug}}}[\ell(\hat{S},E')] \right|.
   $$

   The second term reflects the distribution mismatch between training and test extrusion sequences. By sampling $E'$ from a broad valid space, ESAug enlarges the support of $p_{\text{aug}}$, thereby reducing this mismatch term. As a result, minimizing $L_{\text{ESAug}}$ provides better control over $L_{\text{test}}$, leading to improved generalization.

5. **Random sampling vs local perturbation.**

   ESAug samples from $\mathcal{E}_{valid}$ covering a broader subset of $\mathcal{T}$ than local perturbations. Thus, ESAug should achieve a better performace than local perturbation based on the anaylsis above.

   We compare the two strategies in the table below. Models are implemented with A3PL, ESAug, and GCO. Mean CD is reported here.

   |Method|Random Sampling ↓|Local Perturbation ↓|
   |-|-|-|
   |Text2CAD|22.78|24.34|
   |Drawing2CAD|21.59|23.24|
   |DeepCAD-LFA|21.63|23.08|