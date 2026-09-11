"""Strict validation for the preregistered multi-method survey configuration."""

from __future__ import annotations

from typing import Any


ALLOWED_METHODS = {"ERM", "IRMv1", "VREX", "CORAL", "FISHR", "MLDG", "WEIGHT_NUCLEAR", "FEATURE_NUCLEAR", "SPECTRAL_NORM_REG", "SPECTRAL_REG_2024", "SVB_ORTHDNN", "STABLE_RANK_NORM", "SVD_SPARSE", "SAM", "ASAM", "FAD", "DISAM"}
EXPECTED_BASE = (0.2, 0.1, 0.9, 0.25, 0.25)
EXPECTED_TASK_ID = "TASK-AOPI-SPECTRAL-AND-FLATNESS-PANEL"
FORBIDDEN_SELECTION_KEYS = {
    "best_target", "target_accuracy_grid", "target_hyperparameter_grid",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate scientific semantics and return the same config object."""
    _require(config.get("task_id") == EXPECTED_TASK_ID, "wrong task id")
    methods = config.get("methods")
    _require(isinstance(methods, list) and methods and set(methods) <= ALLOWED_METHODS, "unknown methods")
    _require(len(methods) == len(set(methods)), "duplicate methods")
    candidates = config.get("candidate_methods", methods)
    _require(isinstance(candidates, list) and set(methods) <= set(candidates) <= ALLOWED_METHODS, "bad candidate methods")
    _require("WEIGHT_NUCLEAR" not in methods and "FEATURE_NUCLEAR" not in methods, "legacy rank probes must not be default methods")
    _require(config.get("seeds") == [10, 11, 12, 13, 14], "primary seeds must be 10..14")
    _require(config.get("device") == "cpu", "survey is CPU-only")

    data = config.get("data", {})
    source = tuple(float(x) for x in data.get("source_color_flip_probs", ()))
    base = (
        *source,
        float(data.get("target_color_flip_prob", -1)),
        float(data.get("label_noise", -1)),
        float(data.get("label_noise", -2)),
    )
    _require(base == EXPECTED_BASE, "base world must be (0.2,0.1,0.9,0.25,0.25)")
    _require(data.get("label_rule") == "digit < 5", "wrong label rule")
    _require(data.get("image_subsample") == 2 and data.get("normalize_pixels") is True, "wrong CMNIST preprocessing")

    model = config.get("model", {})
    _require(model == {"input_dim": 392, "hidden_dim": 64, "activation": "relu", "linear_layers": 3}, "wrong model")
    train = config.get("training", {})
    _require(train.get("steps") == 501 and train.get("batch_size_per_environment") == 512, "wrong training schedule")
    _require(float(train.get("learning_rate")) == 0.001 and float(train.get("l2_regularizer_weight")) == 0.001, "wrong optimizer config")

    irm = config.get("irmv1", {})
    _require(irm.get("penalty_anneal_iters") == 100 and float(irm.get("penalty_weight")) == 10000.0, "wrong IRMv1 config")
    vrex = config.get("vrex", {})
    _require(set(vrex) >= {"lambda", "penalty_anneal_iters", "pre_anneal_penalty_weight", "post_anneal_penalty_weight"}, "incomplete V-REx config")
    _require(isinstance(vrex.get("lambda"), (int, float)), "V-REx lambda must be scalar")
    _require(float(vrex["lambda"]) == 10000.0 and vrex["penalty_anneal_iters"] == 100, "wrong V-REx config")
    _require(float(vrex.get("post_anneal_penalty_weight")) == float(vrex["lambda"]), "V-REx post-anneal weight must equal lambda")
    _require(vrex.get("whole_loss_rescale_after_anneal") is True, "V-REx must use official post-anneal whole-loss rescaling")
    coral = config.get("coral", {})
    _require(float(coral.get("gamma")) == 1.0 and coral.get("feature_dimension") == 64, "wrong CORAL config")
    _require(coral.get("covariance_denominator") == "n-1", "CORAL covariance must use n-1")
    fishr = config.get("fishr", {})
    _require(fishr.get("penalty_anneal_iters") == 100, "wrong Fishr anneal")
    _require(float(fishr.get("pre_anneal_penalty_weight")) == 1.0 and float(fishr.get("post_anneal_penalty_weight")) == 10000.0, "wrong Fishr weights")
    _require(fishr.get("whole_loss_rescale_after_anneal") is True and fishr.get("reset_adam_at_anneal") is True, "Fishr must use common anneal/reset/rescale")
    _require(fishr.get("gradient_scope") == "classifier_weight_and_bias" and fishr.get("variance") == "centered_diagonal", "wrong Fishr variance semantics")
    mldg = config.get("mldg", {})
    _require(float(mldg.get("beta")) == 1.0 and float(mldg.get("inner_lr")) == 0.001 and int(mldg.get("inner_steps")) == 1, "wrong MLDG config")
    _require(mldg.get("order") == "first_order" and mldg.get("meta_schedule") == "deterministic_alternating_env0_env1", "wrong MLDG semantics")
    weight_nuclear = config.get("weight_nuclear", {})
    _require(float(weight_nuclear.get("lambda")) == 0.001, "wrong weight nuclear lambda")
    _require(weight_nuclear.get("penalized_layers") == "encoder_linear_only" and weight_nuclear.get("head_penalized") is False, "wrong weight nuclear scope")
    _require(weight_nuclear.get("legacy_only") is True and weight_nuclear.get("default_enabled") is False, "weight nuclear must be legacy-only")
    feature_nuclear = config.get("feature_nuclear", {})
    _require(float(feature_nuclear.get("lambda")) == 0.0001, "wrong feature nuclear lambda")
    _require(feature_nuclear.get("feature_scope") == "concatenated_source_encoder_features", "wrong feature nuclear scope")
    _require(feature_nuclear.get("formula_reference_status") == "VERIFIED_LOSS_PLUS_LAMBDA_SUM_SINGULAR_VALUES", "feature nuclear reference unresolved")
    _require(feature_nuclear.get("legacy_only") is True and feature_nuclear.get("default_enabled") is False, "feature nuclear must be legacy-only")
    snr = config.get("spectral_norm_reg", {})
    _require(float(snr.get("lambda")) == 0.001 and snr.get("penalty") == "sum_sigma1_squared_all_linear_weights", "wrong spectral norm reg config")
    sr2024 = config.get("spectral_reg_2024", {})
    _require(float(sr2024.get("lambda")) == 0.001 and int(sr2024.get("exponent")) == 2 and float(sr2024.get("target_sigma_power")) == 1.0, "wrong SR2024 config")
    svb = config.get("svb_orthdnn", {})
    _require(float(svb.get("svb_factor")) == 0.05 and int(svb.get("projection_frequency")) == 100 and svb.get("projection") == "post_optimizer_step_singular_value_clamp", "wrong SVB config")
    srn = config.get("stable_rank_norm", {})
    _require(float(srn.get("target_rank")) == 8.0 and float(srn.get("spectral_norm_target")) == 1.0, "wrong stable rank norm config")
    svd = config.get("svd_sparse", {})
    _require(svd.get("performance_admission") is False and "SVD_SPARSE_DEFERRED" in svd.get("reason", ""), "SVD sparse must be deferred")
    sam = config.get("sam", {})
    _require(float(sam.get("rho")) == 0.05 and sam.get("adaptive") is False, "wrong SAM config")
    asam = config.get("asam", {})
    _require(float(asam.get("rho")) == 0.5 and float(asam.get("eta")) == 0.01 and asam.get("adaptive") is True, "wrong ASAM config")
    fad = config.get("fad", {})
    _require(fad.get("performance_admission") is False and "FAD_REFERENCE_UNRESOLVED" in fad.get("reason", ""), "FAD must be deferred until exact reference is implemented")
    disam = config.get("disam", {})
    _require(disam.get("performance_admission") is False and "DISAM_REFERENCE_UNRESOLVED" in disam.get("reason", ""), "DISAM must be deferred until exact reference is implemented")
    stable_rank = config.get("stable_rank", {})
    _require(stable_rank.get("registered_as_algorithm") is False and stable_rank.get("diagnostic_only") is True, "stable rank must stay diagnostic-only")
    sweep = config.get("source_only_variant_sweep", {})
    _require(sweep.get("enabled") is True, "source-only variant sweep must be enabled")
    _require(int(sweep.get("calibration_seed")) in {10, 11, 12, 13, 14}, "invalid sweep calibration seed")
    _require(sweep.get("selection_metric") == "source_mean_loss", "variant selection must use source loss")
    _require(sweep.get("tie_breaker") == "source_mean_accuracy", "variant tie-breaker must use source accuracy")
    _require(float(sweep.get("source_accuracy_floor")) == 0.55, "source-only sweep floor must be 0.55")
    _require(sweep.get("canonical_selector") == "best_method_specific_source_geometry_then_source_loss", "wrong source-only selector")
    variants = sweep.get("variants", {})
    expected_sweep_methods = {"SPECTRAL_NORM_REG", "SPECTRAL_REG_2024", "SVB_ORTHDNN", "STABLE_RANK_NORM", "SAM", "ASAM"}
    _require(set(variants) == expected_sweep_methods, "sweep method set is incomplete")
    _require(all(isinstance(values, list) and len(values) >= 3 for values in variants.values()), "each sweep needs at least three variants")
    scalar_methods = expected_sweep_methods - {"SVB_ORTHDNN"}
    _require(all(all(float(value) >= 0.0 for value in variants[method]) for method in scalar_methods), "sweep strengths must be nonnegative")
    _require(all(set(value) == {"svb_factor", "projection_frequency"} and float(value["svb_factor"]) > 0.0 and int(value["projection_frequency"]) >= 1 for value in variants["SVB_ORTHDNN"]), "invalid SVB sweep")

    world = config.get("world", {})
    _require(world.get("dimension") == 5 and world.get("primary_basis") == ["e1", "e2", "e3", "e4", "e5"], "world must be R^5")
    _require(world.get("derived_directions_are_basis") is False, "derived directions cannot be basis coordinates")
    _require(world.get("base_probabilities") == {
        "source_env0_color": 0.2, "source_env1_color": 0.1,
        "evaluation_color": 0.9, "source_label_noise": 0.25,
        "evaluation_label_noise": 0.25,
    }, "base probabilities are not exact")

    response = config.get("response", {})
    _require(float(response.get("delta")) == 0.01 and response.get("horizons") == [1, 5, 20], "wrong response protocol")
    _require(response.get("head_dimension") == 65, "wrong augmented head dimension")
    _require(response.get("source_exposed_basis_indices") == [0, 1, 3], "wrong source-exposed basis")
    selection = config.get("selection", {})
    _require(not (FORBIDDEN_SELECTION_KEYS & set(selection)), "target selection key is forbidden")
    _require(all(value is False for key, value in selection.items() if key.startswith("target_")), "target selection must be false")
    _require(selection.get("response_contrast_promotion") == "descriptive_only", "response contrast cannot promote a verdict")
    grouping = config.get("blind_grouping", {})
    _require(grouping.get("candidate_k") == [2, 3, 4], "cluster candidates are not fixed")
    _require(float(grouping.get("silhouette_threshold")) == 0.25 and float(grouping.get("bootstrap_ari_threshold")) == 0.75, "grouping gates are not fixed")
    _require(grouping.get("semantic_labels_visible_before_freeze") is False, "grouping is not blind")
    diagnostics = config.get("diagnostics", {})
    _require(int(diagnostics.get("source_bank_size_per_environment")) == 64, "wrong diagnostic source bank size")
    _require(int(diagnostics.get("hessian_power_iterations")) >= 3 and int(diagnostics.get("hutchinson_probes")) >= 2, "diagnostic probes too small")
    return config
