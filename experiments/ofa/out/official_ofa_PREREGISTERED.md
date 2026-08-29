# Pre-registered analysis rule — official-pipeline OFA tax (written BEFORE results)

Recorded in advance so the interpretation cannot be fitted to whatever number arrives.

## The fidelity check

The installed ultralytics rejects 9 keys from the released `yolo26s.pt` train_args
(`muon_w`, `sgd_w`, `o2m`, `cls_w`, `topk`, `stride_ratio`, `detach_epoch`, and two
export flags). They were dropped, so this build's defaults apply for those terms. The
supernet's **max arm** runs that same filtered recipe from the same Objects365 init,
so it measures the size of that gap directly.

Released baselines, re-measured under our own val call (published values in brackets):

| model | params | mAP50-95 |
|---|---|---|
| yolo26s | 10.010 M | **0.4714** (0.472) |
| yolo26n | 2.572 M | **0.3949** (0.395) |

## Decision rule

* **max arm ≈ 0.465–0.472** → the dropped keys were immaterial. Tax vs the released
  models is a clean measurement against an officially-converged baseline; report it
  as the headline.
* **max arm ≤ 0.46** → tax vs released **conflates the recipe gap with the cost of
  elasticity** and must be reported as an UPPER BOUND only. The clean number then
  requires a non-elastic control arm: same init, same filtered recipe, 70 epochs, no
  sandwich (~7 h, ~¥56 on the same box). Run it before quoting a tax.
* Either way, report **both** taxes: `released − subnet` (the deployment question)
  and `max arm − subnet` (internally matched, immune to the recipe gap).

## The 0.375 sub-net

No released model exists at width 0.375. Compare it against the released n↔s
interpolation bar, the same method used earlier in this study:
0.3949 @ 2.572 M → 0.4714 @ 10.010 M gives **≈0.429 at 5.88 M**.

## Housekeeping

`last.pt` must be copied off `cocosup` before the instance is released — the results
JSON survives in git, the weights do not.
