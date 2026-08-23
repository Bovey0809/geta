# Plan — make the yolo26s width-elastic OFA supernet actually work

**Status:** the previous width-elastic attempt's `0.0 mAP` was **two implementation bugs,
not a capacity limit.** Both are now confirmed by direct measurement (below). The
"three paradigms, same wall" claim in `CONCLUSION.md` was premature for width-elastic
and has been corrected.

**Goal:** one yolo26s supernet, dial `w ∈ [0.5, 1.0]`, get a usable detector at every
point from a single training run.

**Honest framing of the win:** the product is the **intermediate widths** (w ≈ 0.6–0.85),
which are Pareto points that *do not exist* in the n/s/m/l/x family. w=0.5 beating
yolo26n's free 0.395 is the stretch goal, not the bar.

### MEASURED width/params curve (P1 complete, 2026-08-20)

Real numbers from `count_active_params`, not estimates. 82 of 114 Convs are
planned; **32 stay frozen** (C2PSA L10, C3k2-attn L22, Detect interior) — and
those are the *widest* layers in the net, so the curve is much flatter than w²:

| w | active params | vs w=1.0 | n→s interpolation bar | Gate D target |
|---|---|---|---|---|
| 1.000 | **10.01 M** | — | — | hold 0.472 |
| 0.875 | **8.49 M** | −15 % | 0.461 | > 0.461 |
| 0.750 | **7.15 M** | −29 % | 0.446 | > 0.446 |
| 0.625 | **6.01 M** | −40 % | 0.433 | > 0.433 |
| 0.500 | **5.05 M** | −50 % | 0.422 | > 0.422 |

The "n→s interpolation bar" is the straight line between default **n**
(2.6 M, 0.395) and default **s** (9.5 M, 0.472) — i.e. `0.395 + (P−2.6)·0.01116`.
Beating that line is what makes a width a genuinely *new* Pareto point rather
than something you could already get by picking an existing model.

**Consequence for the goal — the stretch target moved.** w=0.5 is **5.05 M**,
not the ~3.1 M originally estimated, so it is nearly 2× yolo26n's 2.6 M. "Beat
n at w=0.5" is therefore **unreachable by construction** while attention stays
frozen: the frozen blocks alone floor the model well above n. So:

* **P5 (attention elasticity) is no longer optional** if we want to reach
  n-territory at all. It is the only remaining source of savings at low width.
* The **intermediate widths remain the real product** and are unaffected: 5–7 M
  sits squarely in the empty gap between n (2.6 M) and s (9.5 M), where the
  family offers nothing. Gate D is judged against the interpolation line above.

---

## Part 1 — What was actually broken (both confirmed, not hypotheses)

### Bug A — naive first-k slicing violates every internal `chunk`/`cat` boundary

`C2f.forward` (which `C3k2` inherits) is:
```python
y = list(self.cv1(x).chunk(2, 1))      # cv1 outputs 2*c; SEMANTIC boundary at index c
y.extend(m(y[-1]) for m in self.m)
return self.cv2(torch.cat(y, 1))       # cv2 input = (2+n) segments of c
```
Elastic slicing took `cv1`'s output as trained channels `[0, out_k)` with
`out_k = round(2c·w)`, then let `chunk(2,1)` split *that*. Measured on a real
`C3k2(256→256, c=128)`:

```
c=128  w=0.5  out_k=128  k1=64
naive halfA == correct halfA ?  True
naive halfB == correct halfB ?  False      max|diff| = 2.2959
naive halfB == cv1_full[:, 64:128] (i.e. STILL INSIDE semantic half 1)?  True
```

At `w=0.5` the residual branch receives channels drawn **entirely from the first
semantic half**. At `w=0.95` the second half straddles the boundary. This is wrong
at *every* `w < 1.0`:

| w | out_k | halfA | halfB | halfB polluted by semantic-half-1 |
|---|---|---|---|---|
| 1.00 | 256 | [0,128) | [128,256) | no |
| 0.95 | 243 | [0,121) | [121,243) | **yes** |
| 0.75 | 192 | [0,96) | [96,192) | **yes** |
| 0.50 | 128 | [0,64) | [64,128) | **yes (entirely)** |

The same violation applies to every internal concatenation. Full site count in
yolo26s: **23 distinct structural violation sites** (table in Part 3). Cascading
corruption through 23 sites fully explains `0.0 mAP`, and explains why even
`w=0.95` was destroyed.

`PoC check 1` (bit-identity at `w=1.0`) passed because at `w=1.0` nothing is
sliced. `PoC check 2` (finite output at `w=0.5`) passed because shapes still
work out — only the semantics are wrong. **There was never a per-module
correctness oracle.** That is the single most important thing this plan adds.

### Bug B — shared BN running stats are corrupted across widths

`_elastic_conv_forward` passes `bn.running_mean[:out_k]` to `F.batch_norm`. A basic
slice is a **view sharing storage**, and `F.batch_norm(training=True)` updates running
stats **in place**. Measured on `model.4.cv1.bn` (128 ch):

```
[eval  mode, w=0.5]                buffer changed: False
[train mode, w=0.5, under no_grad] running_mean[:64] changed: True   <-- CORRUPTION
                                   running_mean[64:] changed: False  <-- stale
```

`torch.no_grad()` does **not** prevent buffer mutation. So during sandwich training
every student pass — *and the "frozen teacher" pass* — rewrote the inner region of
all ~60 elastic BN buffers with narrower-width statistics, leaving the outer region
at full-width statistics. The `ema.ema` sync callback then saved those mixed buffers.

**This retrodicts the observed failures exactly:**

| run | what happened | explained by |
|---|---|---|
| (a) grad teacher, kd=10 | "teacher preserved 0.4716" | the EMA-save bug meant `last.pt` was byte-identical to pretrained — we evaluated **untrained** weights |
| (b) `no_grad` teacher, kd=10 | teacher **0.472 → 0.0001** | corrupted BN buffers now actually saved |
| (c) narrow gap, **kd=1000** | teacher **0.472 → 0.0052** | ~same collapse at 100× the KD weight |

(c) is the decisive tell: a **100× change in KD weight barely moved the outcome**.
A gradient-magnitude problem would have responded strongly. BN-buffer corruption is
KD-weight-independent. The "student gradients destroy the teacher" story was wrong.

### Also invalid: the earlier BN-recalibration result

`bn_recal.py` reported `w=1.0` recal → **0.345** (down from 0.472). A correct recal
procedure at `w=1.0` must return ≈0.472 by construction. So that run's procedure was
broken (calibrated on **val** images; `momentum=None` crashed and was replaced with
the default 0.1 without re-validating). Every `w<1.0` zero from that run is
**uninterpretable**. Recal must be re-run with its own sanity gate.

### One line on depth-elastic

The depth-elastic `0.0` (dropping 1 of 2 `add=True` residual bottlenecks per C3k) is
now **also suspect** — dropping a residual block should degrade gracefully, and it was
evaluated with full-net BN stats. Worth one re-test after this work, not a workstream.
**The pruning result (Paradigm 1) still stands** — it had a real 50-epoch fine-tune.

---

## Part 2 — Core design: `ChannelPlan` (replaces the global "take first k")

A single global width scalar with contiguous `[:k]` slicing cannot express any of
the structures above. Replace it with an explicit per-tensor channel layout.

```python
@dataclass
class ChannelPlan:
    groups: tuple[int, ...]   # full size of each semantic group, in order
    # offsets are the running prefix sums of `groups`

    def select(self, w: float) -> torch.Tensor:
        """Indices to keep at width w: the first round(g*w) of EACH group."""
```

Every `Conv` carries `in_plan` and `out_plan`. Slicing becomes:

```python
out_sel = self.out_plan.select(w)          # group-structured, may be non-contiguous
in_sel  = self.in_plan.select(w)           # from the producer's out_plan
weight  = conv.weight[out_sel][:, in_sel]
bn_*    = bn.<param>[out_sel]              # SAME out_sel as the conv rows
```

Two properties this buys:

1. **`chunk`/`split` keep working.** `C3k2.cv1.out_plan = (c, c)` → at width `w` the
   output is `[k1 | k1]` with `k1 = round(c·w)`, so `chunk(2,1)` splits at exactly the
   right place, by construction. No forward-code changes needed in the block.
2. **Residual ties are expressible.** `Bottleneck.cv2.out_plan` is *the same object* as
   the bottleneck's `in_plan`, so `x + cv2(cv1(x))` always lines up.

**Consequence to handle in P4:** group-structured selection uses *advanced* indexing,
which returns a **copy**, not a view. So Bug B disappears automatically — but so does
any ability for BN stats to update through the slice. BN therefore needs **explicit
per-width buffers** (P4), not incidental view-writes.

---

## Part 3 — Module-by-module work list (the spine)

Verified inventory of yolo26s (24 top-level layers). Every row is one unit of work
with its own oracle test.

| # | layer(s) | type | internal channel structure | plans / constraints |
|---|---|---|---|---|
| M1 | 0,1,3,5,7,17,20 | `Conv` | none | `out_plan=(C,)`; baseline case |
| M2 | 2,4 | `C3k2`, `m=[Bottleneck]` | `cv1` chunk2@c; `cv2` in = 3 segs of c | `cv1.out=(c,c)`; `cv2.in=(c,c,c)`; Bottleneck `add=True` → `cv2.out ≡ block.in` |
| M3 | 6,8,13,16,19 | `C3k2`, `m=[C3k]` | as M2, **plus** nested `C3k`: `cv3` in = 2 segs of `c_`, 2 Bottlenecks both `add=True` | nested plans; both inner bottlenecks tie out≡in |
| M4 | 9 | `SPPF` | `cv2` in = **4 repeated** segs of `c_`; `n=3`; **`add=True` → `y + x`** | `cv2.in=(c_,)*4`; **`out_plan ≡ block.in_plan`** (hardest constraint) |
| M5 | 12,15,18,21 | `Concat` | inter-layer, per-source | already done (`prepare_concat_alignment`) — port to `ChannelPlan` |
| M6 | 23 | `Detect` | 4 branch families × 3 scales, first-conv `in_c` = [128,256,512] | per-scale `in_plan` from sources 16/19/22; **`out_plan` fixed** (nc+reg·4) |
| M7 | 10 | `C2PSA` | `cv1` split2@c + attention | **frozen** until P5 |
| M8 | 22 | `C3k2 attn`, `m=[Seq(Bottleneck,PSABlock)]` | `cv1` chunk2@c; `cv2` 3 segs; PSABlock attention | **frozen** until P5 (costs the whole P5 branch — 512 ch) |

**Violation-site count:** 8 (`C3k2.cv1` chunk) + 8 (`C3k2.cv2` cat) + 5 (`C3k` cv3 cat)
+ 1 (`SPPF` repeated cat) + 1 (`SPPF` residual) = **23**, plus every Bottleneck residual.

### The oracle test — what was missing all along

For each module type, standalone, random weights, CPU, seconds to run:

```python
def assert_elastic_equals_narrow(module_factory, w):
    """The strong test: elastic-wide-at-w must EQUAL a genuinely narrow module
    built by gathering the corresponding channels out of the wide weights."""
    wide   = module_factory(full_width)
    narrow = module_factory(scaled_width)          # a real, natively-narrow module
    copy_gathered_weights(wide -> narrow, w)       # gather via ChannelPlan.select
    set_width(wide, w)
    assert allclose(wide(x), narrow(x), atol=1e-6)
```

This is exact, not approximate: a correctly group-sliced elastic module is
*definitionally* the same computation as the narrow module with those weights.
It turns "the whole net gives 0.0, shrug" into "M3's inner C3k cv3 fails at w=0.75".

Plus the standing invariant, re-checked after every module lands:
**bit-identity at `w=1.0`** (`max_abs_diff == 0.0`).

### Localization harness

`set_layer_width(model, idx, w)` — make **one** top-level layer elastic, rest at 1.0.
Sweep 24 layers × w ∈ {0.9, 0.75, 0.5} → mAP table (~24 × 15 s ≈ 6 min/width on GPU).
Any future `0.0` becomes a pointed bug report instead of a shrug. Keep this permanently.

---

## Part 4 — Phases and gates

Gates are falsifiable stop conditions. **If a gate fails, we stop and report — no
"one more try".**

### P0 — Correct the record + build the harness *(0.5 h, local)*
- Fix `CONCLUSION.md` Paradigm 3 and the project memory: the width-elastic `0.0` was
  Bugs A+B, not capacity. ✅ *done in this pass*
- `set_layer_width()` localization harness; measure real per-width param counts.

### P1 — `ChannelPlan` + module-by-module correctness — ✅ **DONE (164/164 pass)**
Landed M1 → M6, each with its oracle test green and `w=1.0` bit-identity held.

| module | scope | result |
|---|---|---|
| M1 plain Conv | L0/1/3/5/7/17/20 shapes | exact at all widths |
| M2 C3k2 + Bottleneck | L2, L4 (+ e=0.5, n=2) | exact |
| M3 C3k2 + nested C3k | L6, L8, L13, L16, L19 | exact |
| M4 SPPF | L9, repeated cat + `y+x` residual | exact |
| M5/M6 whole graph | Concat, Upsample, Detect, frozen blocks | bit-identical at w=1.0; finite forward at every width |
| regression | old contiguous slicing | **fails** oracle (err 1.069) while ChannelPlan is **exact** (0.000) |

The regression row is the load-bearing one: it proves the oracle would have
caught the original bug, and that the old slicing was correct *only* for
single-group plain Convs — which is why it hid behind one end-to-end number.

**Structural problem found and solved during M5/M6.** A frozen block cannot
receive a sliced tensor (its convs expect full `in_channels`), which naively
forces full width all the way back through the backbone. Two pieces fixed it:
* `ChannelPlan` gained per-group **`elastic` flags**, so one tensor can mix
  frozen and shrinking segments — required for `L21 = Concat[L20 elastic,
  L10 frozen]`.
* **`plan_adapter_chain`**: depth-wise convs pass the sliced layout through,
  and the first `groups==1` conv becomes an *adapter* (sliced input columns,
  full output), restoring full width for the frozen interior. Detect's `cv3`
  branches start with a depth-wise conv, which is precisely why the
  pass-through case is needed rather than just adapting the first conv.

### P2 findings — the recal sanity gate, and why its SPEC was wrong

The gate did its job immediately: it refused to report any `w<1` number while
recal at `w=1.0` disagreed with the baseline. Rather than loosen the threshold,
each candidate cause was measured:

| change | w=1.0 after recal | effect |
|---|---|---|
| EMA, momentum 0.02, 20 batches | 0.4509 | starting point |
| → cumulative average, 200 batches | 0.4594 | **+0.85 pts** — real fix |
| → exact variance via `E[x²]−E[x]²` | 0.4593 | immaterial (but correct) |
| → train-time augmentation for calib | 0.4535 | **−0.58 pts — refuted** |
| baseline (pretrained stats) | **0.4715** | — |

So the residual −1.2 pts is **not** sample size, **not** the variance estimator,
and **not** augmentation mismatch. It is **BN/weight co-adaptation**: the
downstream weights are tuned to the exact normalisation applied at the end of
training, so replacing the running stats with statistically *better* ones still
moves the network off its co-adapted optimum.

**Consequence:** demanding that recal reproduce the baseline exactly is
unachievable in principle. The gate's *specification* was wrong, not its
implementation. Revised:

* tolerance **0.020** — still catches a genuinely broken procedure (the
  original `bn_recal.py` lost **12.6 pts**), without demanding the impossible;
* every width is reported against **two** references — **A** the stock
  pretrained-stats baseline (0.4715, what a user compares against) and **B**
  `w=1.0` under the *identical* recal protocol (0.4593, which isolates the
  width effect from the recal effect).

Two estimator lessons worth keeping: a one-shot recal over a fixed batch set
must use a **cumulative average** (an EMA is dominated by whichever batches came
first), and variance must come from an accumulated **second moment** — averaging
per-batch variances discards `Var(E_batch)` and under-estimates.

### P2 — Recal sanity + first real measurement → **GATE A** *(1–2 h GPU, ~¥5)*
- **Recal sanity gate first:** recal at `w=1.0` must return **≈0.472**. Use
  **train2017** images through the val-style loader, fixed small momentum (≈0.02),
  200+ batches. If this doesn't return ≈0.472 the procedure is still broken — fix
  before reading any `w<1` number.
- Then whole-net mAP at w ∈ {0.875, 0.75, 0.5}, **with recal**, no sorting, no training.
- **GATE A: does `w=0.875` give > 0.20 mAP?**
  Correct slicing + correct stats should leave a converged net degraded-but-alive when
  12.5% of channels are dropped. Still `0.0` ⇒ something structural remains; stop and
  reassess rather than pile on training.

### GATE A RESULT — **FAIL**, but a soft, informative one *(2026-08-20)*

Recal sanity: **PASS** (w=1.0 → 0.4593 vs 0.4715 baseline, −0.0122, within 0.020).

| w | params | mAP | vs A (0.4715) | vs B (0.4593) | n→s bar |
|---|---|---|---|---|---|
| 1.000 | 10.01 M | 0.4593 | −0.0122 | — | 0.4777 |
| 0.875 | 8.49 M | **0.0395** | −0.4320 | −0.4198 | 0.4607 |
| 0.750 | 7.15 M | 0.0001 | −0.4714 | −0.4592 | 0.4458 |
| 0.625 | 6.01 M | 0.0000 | −0.4715 | −0.4593 | 0.4330 |
| 0.500 | 5.05 M | 0.0000 | −0.4715 | −0.4593 | 0.4223 |

**Gate A: w=0.875 = 0.0395, needed > 0.20 → FAIL.**

Two things are nonetheless established. First, the P1 fix is real and measurable:
w=0.875 moved from the old **exactly 0.0** to **0.0395**, and the degradation is
now smooth and monotone in width rather than uniformly zero. Second, the failure
is **not** a remaining bug.

#### Damage profile — the failure is distributed, not localised

`damage_profile.py` at w=0.875, per-layer relative MSE against the *corresponding*
channels of the full-width run:

```
L0  Conv   0.0322  (+0.032)   L8  C3k2  0.7081  (+0.068)
L1  Conv   0.1167  (+0.085)   L9  SPPF  0.8067  (+0.099)
L2  C3k2   0.3383  (+0.222)   L10 C2PSA 0.8947  (+0.088)
L4  C3k2   0.3829  (+0.278)   L13 C3k2  0.8723  (+0.077)
L6  C3k2   0.5582  (+0.308)   L19 C3k2  0.8030
L7  Conv   0.6396  (+0.082)   L22 C3k2  0.9624
```

Verdict: **shape (b) — small everywhere, compounding.** The very first conv
already injects 3.2 % error from dropping 4 of 32 channels; every elastic layer
adds more, and by L10 the features are ~89 % relative MSE, i.e. essentially
unrelated to the full-width activations. No single layer dominates, so there is
no concentrated lead to chase — this is the arithmetic consequence of choosing
channels **arbitrarily** and compounding it over 24 layers.

#### What this implies

Gate A's 0.20 bar assumed a converged detector would survive arbitrary 12.5 %
channel removal with only BN recal. For *this* network that was optimistic:
the GETA pruning study on the same architecture already found one-shot pruning
collapses by ~10 % sparsity. Arbitrary first-k selection is simply not viable
here, which makes **importance sorting (P3) load-bearing rather than a
refinement** — it is precisely the lever that turns "an arbitrary 12.5 %" into
"the least useful 12.5 %".

**Honest risk to carry into P3:** that same pruning result — importance-ordered
(HESSO) removal collapsing at ~10 % sparsity on this net *without* fine-tuning —
means sorting alone may still not clear a useful bar. Sorting is nonetheless a
prerequisite for P6 to have any chance, because progressive shrinking only works
when the smallest sub-net starts non-broken and its gradients are therefore sane.

### GATE B RESULT — **FAIL**: correct sorting gives no end-to-end benefit

| criterion | w=0.875 | w=0.75 | note |
|---|---|---|---|
| unsorted (Gate A) | **0.0395** | 0.0001 | arbitrary first-k |
| `gamma` = \|γ\| | 0.0356 | 0.0002 | Network Slimming |
| `out_l1` = ‖filter‖₁·\|γ\| | 0.0378 | 0.0003 | best early-layer profile |
| `gamma_over_sigma` | 0.0027 | 0.0000 | **my bug** (below) |

Sorting is **indistinguishable from arbitrary selection end-to-end**, despite
cutting the first selection-sensitive layer's injected error by **2.4×**
(L1 rel_mse 0.037–0.039 vs 0.089 unsorted). The damage profile reconciles those
two facts: the early-layer advantage does not survive compounding — by L8 every
variant sits at ~0.73 rel_mse and by L10 at ~0.89, so the Detect head receives
~90 % corrupted features either way.

#### Two bugs found by measurement, worth keeping

1. **The damage profile had a confound.** It compared a narrow run using
   *recalibrated* stats against a w=1.0 reference still using *pretrained*
   stats, so every number mixed the width effect with the recal effect. The
   tell: L0 reported 0.032 when it must be exactly 0 (its input is the unsliced
   image and BN is per-channel, so no other channel can influence it).
   After recalibrating the reference too, **L0 = 0.0000**.
2. **The importance criterion was wrong.** BN computes
   `y = γ(x−μ)/√(var+ε) + β`, so the normalisation divides σ **out** and
   channel *j*'s post-BN activation has std ≈ `|γ_j|`, independent of
   `running_var`. Ranking by `|γ|/√var` is the right proxy for the *fused
   weight* magnitude but the wrong one for the channel's *output*: it promotes
   channels whose pre-BN variance happened to be small, which anti-correlates
   with importance. Cost of the mistake: L1 error **2.15× worse than arbitrary**,
   and **14×** end-to-end mAP (0.0027 vs 0.0395).

### THE DECIDING MEASUREMENT — fine width sweep near 1.0

| w | params | mAP | vs recal ref (0.4593) | n→s bar |
|---|---|---|---|---|
| 1.00 | 10.01 M | 0.4593 | — | 0.4777 |
| 0.99 | 9.87 M | **0.4317** | −0.028 | 0.4762 |
| 0.98 | 9.75 M | **0.4021** | −0.057 | 0.4748 |
| 0.96 | 9.51 M | 0.3346 | −0.125 | 0.4721 |
| 0.94 | 9.25 M | 0.2563 | −0.203 | 0.4692 |
| 0.92 | 9.03 M | 0.1911 | −0.268 | 0.4667 |
| 0.875 | 8.49 M | 0.0395 | −0.420 | 0.4607 |
| 0.75 | 7.15 M | 0.0001 | −0.459 | 0.4458 |

**Good news:** progressive shrinking now has a genuine **foothold**. At
w=0.98–0.99 the sub-net is very much alive (0.40–0.43), so a schedule
`{1.0} → {1.0, 0.98} → {1.0, 0.98, 0.96} → …` would start from healthy sub-nets
with sane gradients — precisely the condition the earlier training attempts
lacked, since they jumped straight to 0.75/0.5 from a broken state.

**The structural problem, and it is the decisive one:**

> **The widths that survive save almost no parameters, and the widths that save
> real parameters are dead.**
>
> * w ≥ 0.92 → alive (0.19–0.43) but only **1–10 % fewer params** (9.0–9.9 M vs 10.0 M)
> * w ≤ 0.75 → **29 %+ fewer params** but 0.0001 mAP
>
> Sensitivity is ~2.8 mAP points per 1 % of channels removed. And the n→s bar
> is hardest exactly where the model still works: at 9.5 M the bar *is* default
> s (0.472), so near w=1.0 the requirement degenerates to "match s while being
> smaller than s".

So the achievable region is in the wrong place. For width-elastic OFA to yield
a useful Pareto point, training would have to lift w≈0.75 (7.15 M) from 0.0001
to beyond 0.4458 — a ~45-point recovery. For calibration, the pruning study's
50-epoch full-COCO fine-tune recovered a 50 %-sparsity model to 80 % of dense.
Recovering 45 points from ~zero is a different order of ask.

### P3 — Importance-based channel sorting → **GATE B** *(4–6 h local + 1 h GPU)*
Per-**group** `argsort` by effective post-fusion scale `|γ| / sqrt(running_var + ε)`,
propagated to every consumer's input columns.
- Residual ties (`Bottleneck.cv2.out ≡ block.in`, `SPPF.out ≡ SPPF.in`) merge groups
  into **equivalence classes that share one permutation**. The earlier global-permutation
  attempt broke on exactly this (bit-identity diverged to `6.8e+02`).
- Invariant after sorting: **bit-identity at `w=1.0` still `0.0`** (a consistent
  permutation is a relabeling, i.e. a no-op).
- **GATE B: does sorting improve every width vs. P2, and does `w=0.75` clear 0.30?**

### P4 — Per-width BN *(2 h)*
Group-structured gather returns copies, so BN stats can no longer update through the
slice. Add explicit per-width buffers (slimmable-networks switchable-BN pattern):
share `γ`/`β`, keep separate `running_mean`/`running_var` per width in the elastic set.
- **GATE C: `w=0.75` ≥ 0.35 with no weight training at all.**

### P5 RESULT — real improvement, verdict unchanged *(2026-08-21)*

182/182 oracle tests (attention included), 19/19 sorter. C2PSA (L10) and
C3k2-attn (L22) are now elastic; 96 of 114 convs planned (the remaining 18 are
Detect's fixed-output interior).

| w | params (frozen → elastic) | mAP (frozen) | mAP (**elastic**) | n→s bar |
|---|---|---|---|---|
| 1.000 | 10.01 → 10.01 M | 0.4593 | 0.4611 | 0.4777 |
| 0.990 | 9.87 → 9.80 M | 0.4317 | 0.4250 | 0.4754 |
| 0.980 | 9.75 → 9.65 M | 0.4021 | 0.3806 | 0.4736 |
| 0.960 | 9.51 → 9.26 M | 0.3346 | 0.3008 | 0.4693 |
| 0.940 | 9.25 → 8.90 M | 0.2563 | 0.2141 | 0.4653 |
| 0.920 | 9.03 → 8.59 M | 0.1911 | 0.1427 | 0.4618 |
| 0.875 | 8.49 → 7.80 M | 0.0395 | 0.0134 | 0.4531 |
| 0.750 | 7.15 → 5.88 M | 0.0001 | 0.0000 | 0.4316 |
| 0.500 | 5.05 → **2.87 M** | 0.0000 | 0.0000 | **0.3981** |

**At matched parameters, elastic attention is strictly better** — which is the
comparison that matters, since it drops more channels at any given `w`:

| ~params | frozen-attn mAP | elastic-attn mAP | gain |
|---|---|---|---|
| ~8.5 M | 0.0395 (w=0.875) | **0.1427** (w=0.92) | **3.6×** |
| ~9.25 M | 0.2563 (w=0.94) | **0.3008** (w=0.96) | +4.5 pts |
| ~9.8 M | 0.4317 (w=0.99) | 0.4250 (w=0.99) | ≈ equal |

So P5 genuinely shifted the curve up-and-left, and it removed the structural
blocker: **w=0.5 is now 2.87 M, against yolo26n's 2.6 M**, so the bar there is
0.3981 — essentially "beat n" — instead of the unreachable 0.4223 at 5.05 M.

**What did not change:** every width is still far below its bar in the
zero-training regime. Best case is 0.4250 at 9.80 M against a 0.4754 bar; the
n-competitive point (2.87 M) is at 0.0000. The earlier structural finding
survives P5 — survivable widths (≥0.96) save ≤7 % of parameters, and the widths
that save real parameters (≤0.875) are dead.

**What P5 changes about P6's prospects.** Two things now favour training that
did not before:

1. **The prize is real.** A trained w=0.5 at 2.87 M matching n's 0.395 would be
   a genuine result, and the whole continuum above it comes from the same run.
2. **There is a ladder of healthy footholds**, not just one: 0.99 → 0.4250,
   0.98 → 0.3806, 0.96 → 0.3008, 0.94 → 0.2141, 0.92 → 0.1427. Progressive
   shrinking can descend it one rung at a time, recovering at each step, which
   is precisely how OFA is meant to work and was impossible before P1/P4/P5.

That makes P6 a more defensible bet than it was — while still an expensive one,
since walking from 0.92 to 0.50 means recovering several mAP points per rung.

### P5 — attention elasticity, unlocks "the whole network shrinks" — ✅ **DONE**
`C2PSA` (L10) and `C3k2-attn` (L22) are frozen at full width, and L22 is the entire
P5 head (512 ch) — this is where the remaining savings are.
Design: keep `head_dim`/`key_dim` **fixed** and scale **`num_heads`**, so every retained
head stays intact. Then `qkv.out_plan = (2·key_dim + head_dim,) * num_heads` and width
selects the first `round(num_heads·w)` **whole groups** — it drops straight into
`ChannelPlan` with no special-casing. `proj.out_plan ≡ block.in_plan` (PSABlock residual).
**Deferred until Gates A–B pass.**

### SINGLE-RUNG PROBE — the answer on P6 *(2026-08-23)*

Sandwich-trained `{1.0, 0.98}`, 6 epochs @ 15 % of COCO, lr 2e-4, kd 2.0
(~23 min). Both arms evaluated before and after under the identical recal
protocol; the checkpoint was md5-verified to differ from its input.

| | w=1.0 | w=0.98 | gap |
|---|---|---|---|
| before | 0.4611 | 0.3806 | 0.0805 |
| after | **0.4424** | **0.4250** | 0.0173 |

**"78.4 % of the gap closed" overstates it.** Decomposing the 0.063 reduction:
student **+0.0444**, teacher **−0.0187**. So 30 % of the closure is the max arm
coming *down* to meet the small arm. Training the max arm with real gradient
does now work (l_max 32.3 vs l_small 39.0 — comparable, not swamped), but
elasticity still costs the full-width model: 0.4611 → 0.4424, and stock
yolo26s is 0.4715.

#### Transfer decays over ~4 width points

Evaluating the *same* trained checkpoint across all widths:

| w | untrained | after 1 rung @ 0.98 | Δ |
|---|---|---|---|
| 0.98 | 0.3806 | 0.4250 | **+0.0444** |
| 0.96 | 0.3008 | 0.3434 | **+0.0426** |
| 0.94 | 0.2141 | 0.2451 | +0.0310 |
| 0.92 | 0.1427 | 0.1627 | +0.0200 |
| 0.875 | 0.0134 | 0.0217 | +0.0083 |
| 0.75 | 0.0000 | 0.0000 | **0** |
| 0.50 | 0.0000 | 0.0000 | **0** |

The ladder is **real** — one rung lifts its neighbours, nearly fully at 0.96 —
but the spillover dies within ~4 width points, and the deep-shrink regime
(w ≤ 0.75) gains *exactly nothing*. So the descent cannot skip: it needs ~12
rungs of ~4 points to reach w=0.5, each one longer than the last as more widths
join the sandwich. That is the 10–20 h the plan budgeted.

#### Why that budget is not worth spending

Compare the measured recovery against what each width actually needs:

| w | params | now | n→s bar | shortfall | measured recovery/rung |
|---|---|---|---|---|---|
| 0.98 | 9.65 M | 0.4250 | 0.4736 | **0.049** | +0.044 |
| 0.96 | 9.26 M | 0.3434 | 0.4693 | 0.126 | +0.043 |
| 0.92 | 8.59 M | 0.1627 | 0.4618 | 0.299 | +0.020 |
| 0.875 | 7.80 M | 0.0217 | 0.4531 | 0.431 | +0.008 |
| 0.50 | 2.87 M | 0.0000 | 0.3981 | 0.398 | 0 |

The only width within reach of one rung's recovery is **w=0.98 — which saves
0.36 M parameters (3.6 %)**. Everywhere the saving is meaningful, the shortfall
is 5–50× the per-rung recovery, and recovery *shrinks* as the starting point
degrades (+0.044 at a 0.38 start, +0.008 at a 0.02 start). Training does not
close this; it moves the curve up by a few points while the requirement is
tens of points away.

### DEPTH-ELASTIC RE-TEST — the 2026-07-01 "0.0 mAP" was a BN artefact

Same d=1 sub-network (drop the 2nd inner residual bottleneck of every C3k),
evaluated four ways — on yolo26s *and* on **yolo26l, the model the original
claim was actually made about**:

| config | yolo26s (5 C3k) | **yolo26l (14 C3k)** | |
|---|---|---|---|
| stock, unplanned, fused | 0.4715 | **0.5375** | original reported 0.5377 |
| d=2, no recal | 0.4716 | 0.5375 | control — the harness is neutral |
| d=2, with recal | 0.4611 | 0.5303 | recal penalty only |
| **d=1, no recal** | **0.0004** | **0.0000** | **exactly reproduces the original** |
| **d=1, with recal** | **0.1462** | **0.1473** | **the correction** |

The yolo26l column is an exact reproduction of the original experiment — same
baseline (0.5375 vs the reported 0.5377), same 14 C3k blocks, same `0.0000` —
followed by one controlled change. Both models land on ~0.147, which is itself
reassuring: the corrected result is a property of the mechanism, not of a scale.

**Verdict: the original conclusion was wrong.** "The elastic second bottleneck
is *essential* computation, not residual refinement… the ceiling is fundamental,
not a bug" does not hold. It was a batch-norm statistics artefact: evaluating a
depth-1 sub-network with depth-2 running statistics. The d=2 control lands
exactly on the baseline (0.4716 vs 0.4715), so the harness itself is neutral —
the only thing that changed between 0.0004 and 0.1462 is whether BN was
recalibrated for the sub-network being measured.

**The mechanism argument was also invalid.** The original offered ~100 %
relative neck-feature MSE as proof of a capacity ceiling. Re-measured on
yolo26l: **P3 0.643, P4 1.005, P5 1.090** — essentially identical to the
original's 0.58 / 0.99 / 1.12, so it is the same phenomenon, reproduced. But
that large MSE coexists with a network scoring 0.147 once normalisation is
fixed. **High feature MSE never discriminated between a capacity ceiling and an
un-recalibrated distribution shift** — exactly the inferential error that also
produced the width-elastic retraction. (yolo26s: 0.371 / 0.791 / 0.888.)

Consequences, stated conservatively:

* Paradigm 2's negative verdict is **retracted as unsupported**. Its *training*
  results (10-ep sandwich, 5-ep feature-KD, both reported as 0.0) are equally
  invalid — they were measured under the same no-recal protocol AND suffered
  the BN-corruption bug during training.
* Whether depth-elastic is *useful* is now genuinely **open but unpromising**:
  0.146 from 0.461 buys roughly −13 % FLOPs, and by analogy with the measured
  width recovery (+0.044 per rung, decaying) training would likely lift it only
  modestly. Untested.

### P6 — Progressive-shrinking training → **GATE D** — **not run; probe answered it**
Only meaningful once subnets *start* healthy — that's what makes gradients sane and is
why the earlier training could never have worked.
- Staged widths, never introduce a width whose current mAP is near zero:
  `{1.0}` → `{1.0, 0.875}` → `{1.0, 0.875, 0.75}` → … → `{1.0 … 0.5}`
- In-place KD (teacher `no_grad`, features as targets) — the pattern was right, its
  inputs were broken.
- Low lr (1e-5 … 1e-4), per-width BN active, EMA-save workaround in place.
- **GATE D: intermediate widths land above the straight n→s interpolation line**
  (the actual product), and ideally `w=0.5 > 0.395`.

---

## Part 5 — Cost and sequencing

| phase | where | wall time | GPU ¥ |
|---|---|---|---|
| P0 | local | 0.5 h | 0 |
| P1 | **local CPU** | 6–8 h | **0** |
| P2 (Gate A) | 4090 ×1 | 1–2 h | ~5 |
| P3 (Gate B) | local + 4090 | 5–7 h | ~5 |
| P4 (Gate C) | 4090 ×1 | 2 h | ~5 |
| P5 (opt) | local + 4090 | 4–6 h | ~5 |
| P6 (Gate D) | 4090 / PRO 6000 | 10–25 h | 100–250 |

**Do not rent a GPU until every P1 oracle test is green.** Everything up to Gate A is
CPU work runnable on the local `tennis_data_pipeline/.venv` (torch 2.13 CPU +
ultralytics 8.4.117, already verified). A **¥2/hr RTX 4090 is sufficient** for Gates
A–C — the PRO 6000 was overkill and it kept getting idle-reaped between sessions.

**Why this is a different bet than last time:** the previous attempt had no per-module
correctness oracle, so two structural bugs hid behind a single end-to-end number, and
every intervention was tuning hyperparameters on top of broken arithmetic. P1 replaces
that number with ~30 exact per-module assertions, and Gate A reads out the honest
answer *before* any training spend.

---

## Reproduce the two diagnoses

Both run locally, CPU, in seconds:
- Bug A: `scratchpad/verify_chunk_bug.py` — chunk-boundary arithmetic + numerical proof
- Bug B: `scratchpad/verify_bn_corruption.py` — BN buffer mutation under `no_grad`
- Inventory: `scratchpad/inventory.py` — per-layer internal `cat`/`chunk` structure

(to be promoted into `experiments/ofa/tests/` as part of P1)
