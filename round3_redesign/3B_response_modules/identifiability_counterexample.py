"""Executable post-hoc records for the 3B identifiability boundaries."""

from ood_repr_reg.round3r_3b_counterexamples import identical_image_counterexample, partial_overlap_counterexample


if __name__ == "__main__":
    print({"identical_image": identical_image_counterexample(), "partial_overlap": partial_overlap_counterexample()})
