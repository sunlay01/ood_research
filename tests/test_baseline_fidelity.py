import copy
import inspect
from pathlib import Path

import torch
from torch import autograd
from torch.nn import functional as F

from ood_repr_reg.run_task3_baseline_fidelity import _final_verdict, run_equivalence_audit
from ood_repr_reg.task3_baseline_fidelity.ports import (
    BaselineMLP,
    contains_target_reference,
    fish_outer_update,
    full_gradient_parameter_names,
    head_gradient_variance_surrogate_notice,
    head_only_gradient_variance_surrogate,
    iga_objective,
    irmv1_objective,
    irmv1_penalty_from_logits,
    parameter_vector,
    toy_minibatches,
    train_controlled_step,
)
from ood_repr_reg.task3_baseline_fidelity.upstreams import PINNED_UPSTREAMS, upstream_manifest


ROOT = Path(__file__).resolve().parents[1]


def _clone(model):
    return copy.deepcopy(model)


def _manual_bce_erm_update(model, batches, *, lr=1e-3, l2=1e-3):
    all_x = torch.cat([x for x, _ in batches], dim=0)
    all_y = torch.cat([y for _, y in batches], dim=0)
    loss = F.binary_cross_entropy_with_logits(model(all_x), all_y)
    weight_norm = sum(param.norm().pow(2) for param in model.parameters())
    objective = loss + l2 * weight_norm
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt.zero_grad()
    objective.backward()
    opt.step()


def _manual_iga_update(model, batches, *, lr=1e-3, penalty_weight=1000.0):
    params = tuple(model.parameters())
    losses = []
    grads = []
    for x, y in batches:
        logits = model(x)
        env_loss = F.cross_entropy(logits, y.long())
        losses.append(env_loss)
        grads.append(autograd.grad(env_loss, params, create_graph=True, retain_graph=True))
    mean_loss = torch.stack(losses).mean()
    mean_grad = autograd.grad(mean_loss, params, retain_graph=True)
    penalty = next(model.parameters()).new_zeros(())
    for grad in grads:
        for g_env, g_mean in zip(grad, mean_grad):
            penalty = penalty + (g_env - g_mean).pow(2).sum()
    objective = mean_loss + penalty_weight * penalty
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt.zero_grad()
    objective.backward()
    opt.step()


def _manual_fish_update(model, batches, *, lr=1e-3, meta_lr=0.5):
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    inner = copy.deepcopy(model)
    opt = torch.optim.Adam(inner.parameters(), lr=lr)
    for x, y in batches:
        loss = F.cross_entropy(inner(x), y.long())
        opt.zero_grad()
        loss.backward()
        opt.step()
    inner_state = {key: value.detach().clone() for key, value in inner.state_dict().items()}
    model.load_state_dict({key: before[key] + meta_lr * (inner_state[key] - before[key]) for key in before})
    return opt.state_dict()


def test_upstream_manifest_uses_pinned_commits():
    manifest = upstream_manifest(ROOT)
    for key in ("facebook_irm", "domainbed", "fish_author"):
        assert manifest[key]["commit"] == PINNED_UPSTREAMS[key]["commit"]
        if manifest[key]["present"]:
            assert manifest[key]["pinned_commit_matches"] is True
    assert manifest["fish_author_cmnist_status"]["used_for_native_cmnist"] is False


def test_erm_loss_gradient_one_step_matches_upstream_formula():
    torch.manual_seed(1)
    batches = toy_minibatches(output_dim=1, seed=2)
    model = BaselineMLP(output_dim=1)
    reference = _clone(model)
    train_controlled_step(model, batches, method="ERM", step=0, lr=1e-3)
    _manual_bce_erm_update(reference, batches, lr=1e-3)
    assert torch.allclose(parameter_vector(model), parameter_vector(reference), atol=1e-7)


def test_irmv1_penalty_matches_scalar_scale_reference():
    torch.manual_seed(3)
    model = BaselineMLP(output_dim=1)
    x, y = toy_minibatches(output_dim=1, seed=4)[0]
    logits = model(x)
    penalty = irmv1_penalty_from_logits(logits, y)
    scale = torch.tensor(1.0, requires_grad=True)
    manual_loss = F.binary_cross_entropy_with_logits(logits * scale, y)
    manual_grad = autograd.grad(manual_loss, [scale], create_graph=True)[0]
    assert torch.allclose(penalty, manual_grad.square().sum())


def test_irmv1_pre_anneal_update_matches_reference():
    torch.manual_seed(5)
    batches = toy_minibatches(output_dim=1, seed=6)
    model = BaselineMLP(output_dim=1)
    reference = _clone(model)
    row, _ = train_controlled_step(model, batches, method="IRMv1", step=5, lr=1e-3)
    audit = irmv1_objective(reference, batches, step=5, penalty_anneal_iters=100, penalty_weight=10000.0)
    opt = torch.optim.Adam(reference.parameters(), lr=1e-3)
    opt.zero_grad()
    audit.objective.backward()
    opt.step()
    assert row["applied_penalty_weight"] == 1.0
    assert row["rescaled_after_anneal"] is False
    assert torch.allclose(parameter_vector(model), parameter_vector(reference), atol=1e-7)


def test_irmv1_post_anneal_update_uses_rescaled_whole_loss():
    torch.manual_seed(7)
    batches = toy_minibatches(output_dim=1, seed=8)
    model = BaselineMLP(output_dim=1)
    audit = irmv1_objective(model, batches, step=100, penalty_anneal_iters=100, penalty_weight=10000.0)
    unscaled = audit.mean_loss + 1e-3 * audit.l2_weight_norm + 10000.0 * audit.penalty
    assert audit.applied_penalty_weight == 10000.0
    assert audit.rescaled_after_anneal is True
    assert torch.allclose(audit.objective, unscaled / 10000.0)


def test_iga_full_network_gradients_and_default_penalty_match_domainbed_identity():
    torch.manual_seed(9)
    batches = toy_minibatches(output_dim=2, seed=10)
    model = BaselineMLP(output_dim=2)
    audit = iga_objective(model, batches, penalty_weight=1000.0)
    names = full_gradient_parameter_names(model, batches)
    assert set(names) == {name for name, _ in model.named_parameters()}
    assert audit.applied_penalty_weight == 1000.0
    assert torch.isfinite(audit.penalty)


def test_iga_is_not_head_only():
    torch.manual_seed(11)
    batches = toy_minibatches(output_dim=2, seed=12)
    model = BaselineMLP(output_dim=2)
    full = iga_objective(model, batches, penalty_weight=1000.0).penalty
    head_only = head_only_gradient_variance_surrogate(model, batches)
    assert abs(float(full.detach()) - float(head_only.detach())) > 1e-8
    assert any(name.startswith("features") for name in full_gradient_parameter_names(model, batches))


def test_iga_one_step_matches_reference_formula():
    torch.manual_seed(13)
    batches = toy_minibatches(output_dim=2, seed=14)
    model = BaselineMLP(output_dim=2)
    reference = _clone(model)
    train_controlled_step(model, batches, method="IGA", step=0, lr=1e-3)
    _manual_iga_update(reference, batches, lr=1e-3, penalty_weight=1000.0)
    assert torch.allclose(parameter_vector(model), parameter_vector(reference), atol=1e-7)


def test_fish_inner_clone_sequential_update_and_outer_interpolation_match_reference():
    torch.manual_seed(15)
    batches = toy_minibatches(output_dim=2, seed=16)
    model = BaselineMLP(output_dim=2)
    reference = _clone(model)
    audit = fish_outer_update(model, batches, lr=1e-3, meta_lr=0.5)
    state = _manual_fish_update(reference, batches, lr=1e-3, meta_lr=0.5)
    assert audit.parameter_delta_norm > 0.0
    assert audit.optimizer_inner_state["state"]
    assert state["state"]
    assert torch.allclose(parameter_vector(model), parameter_vector(reference), atol=1e-7)


def test_fish_carries_inner_optimizer_state():
    torch.manual_seed(17)
    batches = toy_minibatches(output_dim=2, seed=18)
    model = BaselineMLP(output_dim=2)
    first = fish_outer_update(model, batches, lr=1e-3, meta_lr=0.5)
    second = fish_outer_update(model, batches, lr=1e-3, meta_lr=0.5, optimizer_inner_state=first.optimizer_inner_state)
    assert first.optimizer_inner_state["state"]
    assert second.optimizer_inner_state["state"]
    assert second.parameter_delta_norm > 0.0


def test_no_target_data_enters_baseline_training_or_selection_code():
    source = inspect.getsource(train_controlled_step)
    assert not contains_target_reference(source)
    assert "target" not in source.lower()


def test_old_head_gradient_surrogate_cannot_be_reported_as_iga_or_fish():
    notice = head_gradient_variance_surrogate_notice()
    assert notice["method_label"] == "HEAD_GRADIENT_VARIANCE_SURROGATE"
    assert notice["not_a_reproduction_of_iga_or_fish"] is True
    assert "head-only" in str(notice["why_not_iga"])
    assert "no inner clone" in str(notice["why_not_fish"])


def test_equivalence_audit_drives_verdict_gate():
    equivalence = run_equivalence_audit()
    verdict = _final_verdict(
        irm_status={"passed": True},
        iga_native=[{"status": "completed"}],
        fish_native=[{"status": "completed"}],
        equivalence=equivalence,
    )
    assert verdict == "BASELINE-FIDELITY-PASS"
    failed = _final_verdict(
        irm_status={"passed": False},
        iga_native=[{"status": "completed"}],
        fish_native=[{"status": "completed"}],
        equivalence=equivalence,
    )
    assert failed == "BASELINE-FIDELITY-FAIL"
