import pytest

from ood_repr_reg.certificates import (
    CrossDomainCertificate,
    metric_root,
    positive_degradation,
    source_fit_excess,
)


def test_positive_degradation_is_directional() -> None:
    assert positive_degradation(1.5, 1.0) == pytest.approx(0.5)
    assert positive_degradation(0.8, 1.0) == 0.0


def test_source_fit_excess_uses_erm_reference() -> None:
    assert source_fit_excess(0.4, 0.25) == pytest.approx(0.15)
    assert source_fit_excess(0.2, 0.25) == 0.0


def test_metric_root_rejects_invalid_penalties() -> None:
    assert metric_root(0.25) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="non-negative"):
        metric_root(-1.0)


def test_certificate_separates_inputs_from_target_evaluation() -> None:
    certificate = CrossDomainCertificate(
        algorithm="coral",
        source_risk=0.4,
        erm_source_risk=0.25,
        original_metric=0.09,
        target_risk=1.4,
        erm_target_risk=1.0,
        source_domain_spread=0.3,
        conditional_label_shift=0.6,
        representation_insufficiency=0.7,
        target_coverage_residual=0.8,
    )

    assert certificate.target_degradation == pytest.approx(0.4)
    assert certificate.source_only_inputs()["source_fit_excess"] == pytest.approx(0.15)
    assert "target_risk" not in certificate.source_only_inputs()
    assert "erm_target_risk" not in certificate.source_only_inputs()
    assert "target_coverage_residual" not in certificate.source_only_inputs()
    assert "conditional_label_shift" not in certificate.source_only_inputs()
    assert "representation_insufficiency" not in certificate.source_only_inputs()
    assert "target_coverage_residual" not in certificate.metric_views()["source_only_observable"]
    assert "conditional_label_shift" not in certificate.metric_views()["source_only_observable"]
    assert certificate.as_record()["conditional_label_shift"] == pytest.approx(0.6)
    assert certificate.as_record()["target_coverage_residual"] == pytest.approx(0.8)


def test_certificate_requires_finite_nonnegative_metric() -> None:
    with pytest.raises(ValueError, match="finite"):
        CrossDomainCertificate("erm", 0.1, 0.1, float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        CrossDomainCertificate("erm", 0.1, 0.1, -0.1)
