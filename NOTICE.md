# Provenance and Dependencies

The model is reorganized from the corrected full DNU-SS research implementation.
The mathematical Gabor, ECA, ArcFace, spatial-indexed LIF, SpikingSSA and
SpikingMLP operations were retained while their file/module boundaries were
made consistent with the manuscript. Other complete-model operations were
written directly in paper-named modules and checked against existing weights.

The release depends on external PyTorch, torchvision, SpikingJelly, CuPy
(optional), NumPy, SciPy, scikit-learn, Pillow and Matplotlib packages.
Their respective licenses remain applicable. No third-party package source is
vendored. Inspect the original project's provenance and attribution obligations
before selecting a redistribution license for this repository.

No open-source license is asserted by this notice. Ownership, author metadata
and the final repository license must be confirmed by the project owner before
public distribution. No dataset images, pretrained weights or manuscript PDF
are distributed in this export.
