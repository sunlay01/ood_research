"""Strict validation for the preregistered multi-method survey configuration."""

from __future__ import annotations

from typing import Any


ALLOWED_METHODS = {"ERM", "IRMv1", "VREX", "CORAL"}
EXPECTED_BASE = (0.2, 0.1, 0.9, 0.25, 0.25)
FORBIDDEN_SELECTION_KEYS = {
    "best_target", "target_accuracy_grid", "target_hyperparameter_grid",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate scientific semantics and return the same config object."""
    _require(config.get("task_id") == "TASK-AOPI-MULTIMETHOD-MECHANISM-SURVEY", "wrong task id")
    methods = config.get("methods")
    _require(isinstance(methods, list) and methods and set(methods) <= ALLOWED_METHODS, "unknown methods")
    _require(len(methods) == len(set(methods)), "duplicate methods")
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
    return config
