# Model card

Model: Gemino reproduction, needle_drop personalized, step30500. Publisher: Pengfei Yin, Tianjin University. Intended use: noncommercial research in reference-conditioned reconstruction. Not an official Gemino checkpoint.

Architecture: official3-pathways, HR skips,10 keypoints with Jacobian, decoded256x256 low input,1024x1024 target. FOMM initialization with official exclusions. Training: batch2, Adam0.0002, repeats15,30epochs39360updates; released candidate selected from evaluated checkpoints. Training13source videos29m10s, validation2sources4m, test5sources17m15s. Source-level isolation; final test-set model acceptance has not been performed.

Selection: evaluated3000/10000/20500/30500/39360.30500 LPIPS: low-head critical0.2743287, window0.2489879, extra6clips0.2610669.10000 had higher extra6 SSIM. These validation-based scores do not establish an optimum among all intermediate steps. Preserve reference, codec, crop and frame rules for comparisons.

Codec: original modified aiortc VP8/libvpx, nearest downsample,45k label mapped internally to200k. Validation payload195.67/195.85Kbps, excluding references and network overhead. Training creates fresh codec per target frame; evaluation uses continuous codec. Runtime differs from historical binaries. No claim of45Kbps performance or paper-level rate-distortion results.

Limitations: single identity; cropped square videos; loss of fine texture; possible temporal artifacts, occlusion errors and domain shift. Cannot infer support for arbitrary identities/resolutions or truthful recovery of unreadable details. Use original video for evidentiary detail.

Weights: inference-only generator, keypoint detector, config, global_step; no optimizer. SHA256 in model_manifest.json. Inference uses torch.load(weights_only=True) and strict parameter loading. Use matching configuration. Noncommercial research under applicable upstream terms; retain licenses. Training-source rights and personal likeness rights are not granted by this repository.
