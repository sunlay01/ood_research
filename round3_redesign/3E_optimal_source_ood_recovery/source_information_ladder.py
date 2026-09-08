"""Reproducible entry point for the canonical 3E information ladder."""

from ood_repr_reg.round3r_3e_information_ladder import information_ladder, monotonicity_audit


if __name__ == "__main__":
    rows = information_ladder()
    print(rows)
    print(monotonicity_audit(rows))
