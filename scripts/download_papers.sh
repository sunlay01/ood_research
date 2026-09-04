#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
paper_dir="$repo_root/papers/pdfs"
mkdir -p "$paper_dir"

download_one() {
  local file="$1"
  local url="$2"
  local target="$paper_dir/$file"
  if [[ -s "$target" && "${FORCE_DOWNLOAD_PAPERS:-0}" != "1" ]]; then
    return 0
  fi
  curl -fL --retry 2 --max-time 120 -A "Mozilla/5.0" "$url" -o "$target"
}

download_one "representation_bayesian_risk_decompositions_2020_wu.pdf" "https://arxiv.org/pdf/2004.10390"
download_one "failure_modes_domain_generalization_algorithms_2022_galstyan.pdf" "https://openaccess.thecvf.com/content/CVPR2022/papers/Galstyan_Failure_Modes_of_Domain_Generalization_Algorithms_CVPR_2022_paper.pdf"
download_one "representation_regularization_invariance_dg_2022_shui.pdf" "https://link.springer.com/content/pdf/10.1007/s10994-021-06080-w.pdf"
download_one "domain_generalization_without_excess_empirical_risk_2022_sener.pdf" "https://proceedings.neurips.cc/paper_files/paper/2022/file/57568e093cbe0a222de0334b36e83cf5-Paper-Conference.pdf"
download_one "understanding_hessian_alignment_dg_2023_hemati.pdf" "https://openaccess.thecvf.com/content/ICCV2023/papers/Hemati_Understanding_Hessian_Alignment_for_Domain_Generalization_ICCV_2023_paper.pdf"
download_one "invariant_risk_minimization_total_variation_model_2024_lai.pdf" "https://raw.githubusercontent.com/mlresearch/v235/main/assets/lai24c/lai24c.pdf"
download_one "moment_alignment_gradient_hessian_matching_dg_2025_chen.pdf" "https://raw.githubusercontent.com/mlresearch/v286/main/assets/chen25f/chen25f.pdf"
download_one "bridging_domain_invariance_diversity_2026_wang.pdf" "https://www.jmlr.org/papers/volume27/25-0399/25-0399.pdf"
download_one "landau_theory_invariant_learning_2026_wang.pdf" "https://arxiv.org/pdf/2608.09396"
download_one "implicit_differentiation_lasso_hyperparameter_2020_bertrand.pdf" "https://proceedings.mlr.press/v119/bertrand20a/bertrand20a.pdf"

if command -v pdfinfo >/dev/null 2>&1; then
  for file_path in "$paper_dir"/*.pdf; do
    pdfinfo "$file_path" | awk -v file="$(basename "$file_path")" '/^Pages:/ {print file "\t" $2 " pages"}'
  done
else
  file "$paper_dir"/*.pdf
fi
