=== Sanity Check (g_t vs g_0) ===
Readout method: mean
Mean ||g(H') - g(H)|| over first train batch: 0.000013
(If this is ~0 for mean readout, diffusion can be 'washed out' by pooling.)
Evaluating on test set...
Test Accuracy: 0.5397

=== Diagnostics Notice ===
This repo currently has no explicit cache in graph/diffusion/readout/backbone (no memoization found).
So there's nothing to 'disable' unless you added caching elsewhere.

=== Single-Image Diagnostics (Norms/Logits/W stats) ===
Fixed input: test_dataset[0] (fallback to 0 if OOB)
Baseline: t=0, sigma_s=0.1, sigma_f=0.8
||H'(t=0)-H||: 0.000000 (should be ~0)
||g'(t=0)-g||: 0.000000

=== W Stats (sigma=cfg) ===
all:    mean=0.00509867 std=0.071199 entropy=5.27812
offdiag: mean=1.10472e-27 std=0 entropy=1.05702e-06

=== Diag (sigma=cfg, t=0.0) ===
||H'(t)-H||: 0.000000
||g - mean(H)|| (ref): 0.000000
vs baseline(t=0,sigma=cfg): ||H'-H'0||=0.000000, ||g-g0||=0.000000, ||logits-logits0||=0.000000

=== Diag (sigma=cfg, t=10.0) ===
||H'(t)-H||: 0.010559
||g - mean(H)|| (ref): 0.000394
vs baseline(t=0,sigma=cfg): ||H'-H'0||=0.010559, ||g-g0||=0.000394, ||logits-logits0||=0.000006

=== Diag (sigma=cfg, t=50.0) ===
||H'(t)-H||: 0.052798
||g - mean(H)|| (ref): 0.001972
vs baseline(t=0,sigma=cfg): ||H'-H'0||=0.052798, ||g-g0||=0.001972, ||logits-logits0||=0.000028

=== Diag (sigma=cfg, t=100.0) ===
||H'(t)-H||: 0.105596
||g - mean(H)|| (ref): 0.003944
vs baseline(t=0,sigma=cfg): ||H'-H'0||=0.105596, ||g-g0||=0.003944, ||logits-logits0||=0.000055

=== W Stats (sigma=1e-6) ===
all:    mean=0.0031237 std=0.0558027 entropy=4.78749
offdiag: mean=0 std=0 entropy=nan

=== Diag (sigma=1e-6, t=0.0) ===
||H'(t)-H||: 0.000000
||g - mean(H)|| (ref): 0.000000
vs baseline(t=0,sigma=cfg): ||H'-H'0||=0.000000, ||g-g0||=0.000000, ||logits-logits0||=0.000000

=== Diag (sigma=1e-6, t=10.0) ===
||H'(t)-H||: 600.040039
||g - mean(H)|| (ref): 14.128074
vs baseline(t=0,sigma=cfg): ||H'-H'0||=600.040039, ||g-g0||=14.128074, ||logits-logits0||=0.230135

=== Diag (sigma=1e-6, t=50.0) ===
||H'(t)-H||: 600.067261
||g - mean(H)|| (ref): 14.129642
vs baseline(t=0,sigma=cfg): ||H'-H'0||=600.067261, ||g-g0||=14.129642, ||logits-logits0||=0.230155

=== Diag (sigma=1e-6, t=100.0) ===
||H'(t)-H||: 600.067261
||g - mean(H)|| (ref): 14.130794
vs baseline(t=0,sigma=cfg): ||H'-H'0||=600.067261, ||g-g0||=14.130794, ||logits-logits0||=0.230168

=== W Stats (sigma=1e6) ===
all:    mean=1 std=0 entropy=10.5562
offdiag: mean=1 std=0 entropy=10.5511

=== Diag (sigma=1e6, t=0.0) ===
||H'(t)-H||: 0.000000
||g - mean(H)|| (ref): 0.000000
vs baseline(t=0,sigma=cfg): ||H'-H'0||=0.000000, ||g-g0||=0.000000, ||logits-logits0||=0.000000

=== Diag (sigma=1e6, t=10.0) ===
||H'(t)-H||: 822.219543
||g - mean(H)|| (ref): 0.000090
vs baseline(t=0,sigma=cfg): ||H'-H'0||=822.219543, ||g-g0||=0.000090, ||logits-logits0||=0.000001

=== Diag (sigma=1e6, t=50.0) ===
||H'(t)-H||: 822.256836
||g - mean(H)|| (ref): 0.001510
vs baseline(t=0,sigma=cfg): ||H'-H'0||=822.256836, ||g-g0||=0.001510, ||logits-logits0||=0.000021

=== Diag (sigma=1e6, t=100.0) ===
||H'(t)-H||: 822.256897
||g - mean(H)|| (ref): 0.003022
vs baseline(t=0,sigma=cfg): ||H'-H'0||=822.256897, ||g-g0||=0.003022, ||logits-logits0||=0.000042

=== Requested Pairwise Diffs ===
[t=0 vs t=10 @ sigma=cfg]
||H'(10)-H'(0)||: 0.010559
||g(10)-g(0)||: 0.000394
||logits(10)-logits(0)||: 0.000006
[sigma=1e-6 vs sigma=1e6 @ t=10]
||H'(large)-H'(small)||: 716.790833
||g(large)-g(small)||: 14.128162
||logits(large)-logits(small)||: 0.230136