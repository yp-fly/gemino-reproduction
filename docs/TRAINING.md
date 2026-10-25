# Reproduction notes

Original snapshots and hashes: provenance/source_provenance.json. The vendored model and codec sources are unchanged; added package initialization and runtime adapters are separate. Original dataset crop annotations and final author weights were unavailable. Videos are timestamp-selected from native sources according to provenance/training_manifest.json; crop annotations and val split differ from author processing. Frames sampled independently from clips of the same person; official augmentation and config applied. torchvision/skimage compatibility pad alias and Gaussian global cache reset are in formal_compat.py.

Fresh initialization for the aligned formal run differs from earlier exploratory shape-matching trials. Do not combine old trial metrics with this release. Current torch2.5 default grid_sample False matches repository torch1.6 default. Old environments and exact libvpx binary were not replicated. Weights imported by shape and policy are described in reproduction/formal_compat.py. Training requires independently obtained source videos/FOMM/VGG; those files are not included.

The selected inference candidate is30500, not the final39360 optimizer checkpoint. Further training from it requires reinitializing or separately obtaining optimizer state; it is sufficient for standalone inference.
