import numpy as np

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.blind_grouping import adjusted_rand_index, blind_group
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.signatures import mechanism_signature_rows


def signatures():
    return [
        {"opaque_direction_id": f"u{index:03d}", "x": float(index // 4), "y": float(index % 4) * 0.01}
        for index in range(11)
    ]


def test_row_permutation_preserves_grouping_partition():
    original = blind_group(signatures(), repeats=5)
    permuted_rows = list(reversed(signatures()))
    permuted = blind_group(permuted_rows, repeats=5)
    by_id = dict(zip(original["opaque_ids"], original["labels"]))
    restored = np.asarray([by_id[item] for item in permuted["opaque_ids"]])
    assert adjusted_rand_index(restored, permuted["labels"]) == 1.0


def test_grouping_inputs_are_opaque_and_do_not_need_semantics_or_target_accuracy():
    result = blind_group(signatures(), repeats=5)
    assert result["status"] in {"STAGE4-PASS", "STAGE4-NO-STABLE-GROUPING"}
    assert all(item.startswith("u") for item in result["opaque_ids"])
    assert "target_acc" not in result["feature_keys"]


def test_signature_columns_use_opaque_method_codes():
    a = [{"opaque_direction_id": "u000", "method": "ERM", "A_normalized_norm": 1.0}]
    o = [{"opaque_direction_id": "u000", "method": "ERM", "O_normalized_norm": 2.0}]
    response = [{"opaque_direction_id": "u000", "method": "ERM", "K": 1, "normalized_source_response": 3.0}]
    row = mechanism_signature_rows(a, o, response)[0]
    assert row == {"opaque_direction_id": "u000", "a_m000": 1.0, "o_m000": 2.0, "r_m000_K1": 3.0}
