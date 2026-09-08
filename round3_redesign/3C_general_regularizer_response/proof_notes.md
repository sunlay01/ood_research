# Proof and audit notes

The source objective is evaluated directly from stacked population moments. The
autodiff Jacobians and central differences are independent checks of the same
first-order condition. A finite-dimensional same-`(b,K)`/different-`Pi` fixture
is emitted to prevent frozen curvature from being interpreted as source
adaptation.
