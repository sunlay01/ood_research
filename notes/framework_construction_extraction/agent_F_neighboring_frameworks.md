# Agent F — Neighboring formal methods

## External search and selected works

Crossref searches also returned *Generalization Bounds: Perspectives from Information Theory and PAC-Bayes* (2025), DOI [10.1561/2200000112](https://doi.org/10.1561/2200000112), and *Partial Transportability for Domain Generalization* (2024), DOI [10.52202/079017-4376](https://doi.org/10.52202/079017-4376). I retain only the construction lessons that overlap mature information/transport frameworks rather than treating these as proof of a new OOD object.

## Framework cards

### Information-theoretic generalization
- **Scientific uncertainty:** learner-data dependence and stochastic information leakage.
- **Primitive:** mutual information or information density.
- **Intermediate:** information radius controlling expected generalization.
- **Backbone:** mutual-information/change-of-measure inequality plus concentration.
- **Cost:** often distribution- and learner-specific; an information term does not identify target shifts.

### Partial transportability
- **Scientific uncertainty:** only some causal effects or components transport.
- **Primitive:** selection/transport graph and partial query.
- **Intermediate:** identifiable subset and residual nontransportability.
- **Backbone:** partial identification bounds rather than a forced point estimate.
- **Lesson:** a framework can be useful while returning an interval when point identification fails.

## Cross-field summary

Recurring architecture patterns: define the uncertainty/query; introduce a certificate or interval; prove a generic theorem; retain non-identifiability. Distinctive pattern: partial identification is an honest output, not a failure of notation. Primitive choice follows the downstream decision/query. For OOD, transfer this willingness to return bounds/intervals rather than inventing a point target.
