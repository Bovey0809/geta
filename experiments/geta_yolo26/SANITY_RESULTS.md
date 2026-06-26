# GETA self-validation: do its OWN example models prune correctly?
21 sanity tests (build -> OTO -> construct_subnet -> assert output diff < 1e-4 + size reduction).
NOTE: examples don't run out-of-the-box. Required fixes: uncomment backends/__init__.py exports
(all were commented), torch-2.8 (torch.load weights_only, _optimize_graph shim), graphviz no-op, einops.

## Clean PASS (output diff ~1e-9, structurally-lossless prune):
resnet18 -72% | resnet50 -81% | resnet18-DuBN -57% | vgg16bn -93% | densenet121 -69% |
concat_case1 -40% | concat_case2 -91% | groupconv -29% | convtranspose_in_case1 -74% |
convtranspose_in_case2 -83% | instance-norm(in_case3) -80% | weight_share_case1 -83% |
weight_share_case2 -47% | batchnorm -55%   => 14/21 clean

## BROKEN / suspect:
- GroupNorm: case1 RuntimeError, case3 ZeroDivisionError, case4 "num groups -64"; case2 prunes -1% (nothing).
- SimpleViT: TIMEOUT (>240s) -- transformer graph construct hangs.
- ConvNeXt-tiny: test "PASS" but output diff=0.549 (>>1e-4) -- construct does NOT preserve output (loose test).
- ResNet34-DuBIN: ValueError in construct.

## Verdict
GETA reliably prunes the CNN family smaller (40-93%) with output preserved (diff ~1e-9). It BREAKS on
GroupNorm and Transformers/attention, and silently mis-prunes ConvNeXt. This is consistent with our
YOLO26 experience (attention C2PSA + concat-heavy = the flaky structures). These tests verify
structural output-preservation (foundation for maintaining accuracy), not accuracy-after-training.
