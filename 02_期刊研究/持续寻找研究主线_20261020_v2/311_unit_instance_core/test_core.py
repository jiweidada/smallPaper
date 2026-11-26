"""Parent-run CPU-only synthetic contracts. This file never loads real inputs/CPs.

Example: D:\\Anaconda\\python.exe -s -X utf8 -B test_core.py --output result.json
All output creation is exclusive: use a new filename for a subsequent run.
"""
import os
import sys
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import hashlib
import json
import pathlib
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
RUNTIME = HERE.parents[1] / "code/dn_pre_frozen_diagnostic_20261020_v1/runtime"
assert sys.flags.no_user_site, "Run python -s to isolate user site packages"
assert RUNTIME.is_dir(), RUNTIME
sys.path[:0] = [str(HERE), str(RUNTIME)]  # Dependency runtime only; no native/source imports.
import psutil
import torch
from core import (CLASSES, DIM, SIDE, CompletionHead, copy_common, density_inputs,
                  independent_cap, parameter_count)

COUNTS = dict(synthetic_head_builds=0, synthetic_head_forwards=0,
              synthetic_autograd_calls=0, optimizer_steps=0,
              real_image_reads=0, checkpoint_loads=0, native_imports=0,
              global_detector_forwards=0, heldout_reads=0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resources():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    proc = psutil.Process()
    assert {13, 15}.issubset(proc.cpu_affinity()), proc.cpu_affinity()
    proc.cpu_affinity([13, 15])
    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    assert sorted(proc.cpu_affinity()) == [13, 15]
    assert proc.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
    assert torch.version.cuda is None and not torch.cuda.is_available()
    assert psutil.virtual_memory().available / 2**30 > 8
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(2026102104)
    return dict(threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
                affinity=proc.cpu_affinity(), priority="BelowNormal",
                free_RAM_GiB=psutil.virtual_memory().available / 2**30,
                torch=torch.__version__, torch_file=torch.__file__,
                executable=sys.executable, no_user_site=bool(sys.flags.no_user_site),
                MKL_THREADING_LAYER=os.environ["MKL_THREADING_LAYER"], mha_fastpath=False)


def close(a, b, atol=1e-12):
    torch.testing.assert_close(a, b, atol=atol, rtol=0.)


def grads(loss, inputs):
    COUNTS["synthetic_autograd_calls"] += 1
    return torch.autograd.grad(loss, inputs, allow_unused=True)


def head_call(model, data, candidates, rgb=False):
    COUNTS["synthetic_head_forwards"] += 1
    return model(data, *candidates) if rgb else model.forward_features(data, *candidates)


def operator_contracts():
    m = torch.tensor([[[.2, 0., .8, .1], [.3, .4, .1, .2], [.2, .2, .2, .2]],
                      [[0., 0., 0., 0.], [.1, .2, .3, .4], [.2, .2, .2, .2]]], dtype=torch.float64)
    cost = torch.randn(2, 4, 5, dtype=torch.float64)
    null = torch.randn(2, 5, dtype=torch.float64)
    cls = torch.tensor([[0, 0, 1, 2, -17], [0, 1, 2, -17, -17]])
    valid = cls >= 0
    out = independent_cap(m, cost, null, cls, valid)
    close(out["p"].sum(1) + out["p0"], valid.double())
    close(out["T"].sum(1) + out["z"], valid.double())
    assert bool((out["explained"] <= m + 1e-12).all())
    assert bool((out["z"] >= 0).all()) and bool((out["residual"] >= 0).all())
    close(out["residual"], (m - out["u"]).clamp_min(0.))
    assert all(bool(torch.isfinite(v).all()) for v in out.values())
    # All-zero mass admits only null, not NaNs or a uniform visual explanation.
    zero = independent_cap(torch.zeros_like(m), cost, null, cls, valid)
    close(zero["z"], valid.double())
    close(zero["T"], torch.zeros_like(cost))
    empty = independent_cap(m, cost[..., :0], null[:, :0], cls[:, :0], valid[:, :0])
    close(empty["residual"], m)
    assert empty["T"].shape == (2, 4, 0)
    return dict(column_conservation=True, row_capacity=True, clip_identity=True,
                zero_mass_null_only=True, empty_supply_identity=True)


def hand_and_gradient_contracts():
    m = torch.tensor([[[.4, .4, .35]]], dtype=torch.float64)
    target_p = torch.tensor([.8, .1, .05], dtype=torch.float64)
    cost = -(target_p / m.flatten()).log().reshape(1, 3, 1)
    null = -torch.tensor([[.05]], dtype=torch.float64).log()
    cls, valid = torch.zeros((1, 1), dtype=torch.long), torch.ones((1, 1), dtype=torch.bool)
    out = independent_cap(m, cost, null, cls, valid)
    close(out["p"].flatten(), target_p)
    close(out["T"].flatten(), torch.tensor([.4, .1, .05], dtype=torch.float64))
    close(out["z"], torch.tensor([[.45]], dtype=torch.float64))
    close(out["residual"].flatten(), torch.tensor([0., .3, .3], dtype=torch.float64))
    # A genuine strict-overload counterexample: both T and null cease to depend on C/kappa.
    c = torch.zeros((1, 1, 1), dtype=torch.float64, requires_grad=True)
    k = torch.full((1, 1), 4., dtype=torch.float64, requires_grad=True)
    tiny = independent_cap(torch.full((1, 1, 1), .1, dtype=torch.float64), c, k, cls, valid)
    assert tiny["u"].item() > .1
    gc, gk = grads(-tiny["T"].log().sum() - tiny["z"].log().sum(), (c, k))
    close(gc, torch.zeros_like(gc))
    close(gk, torch.zeros_like(gk))
    # Same-cost cell splitting preserves mass-reference allocation after aggregation.
    mm = torch.tensor([[[.7, 1.2]]], dtype=torch.float64)
    cc = torch.tensor([[[.3], [-.2]]], dtype=torch.float64)
    kk = torch.tensor([[.1]], dtype=torch.float64)
    first = independent_cap(mm, cc, kk, cls, valid)
    split = independent_cap(torch.tensor([[[.7, .6, .6]]], dtype=torch.float64),
                            torch.tensor([[[.3], [-.2], [-.2]]], dtype=torch.float64), kk, cls, valid)
    for name in ("p", "T"):
        close(first[name][:, 0], split[name][:, 0])
        close(first[name][:, 1], split[name][:, 1:].sum(1))
    close(first["z"], split["z"])
    close(first["residual"][:, :, 1], split["residual"][:, :, 1:].sum(2))
    return dict(hand_cap_and_return_null=True, strict_overload_zero_cost_gradient=True,
                equality_subgradient_not_assumed=True, mass_reference_split_invariance=True)


def fixtures():
    f = torch.randn(2, DIM, SIDE, SIDE)
    boxes = torch.tensor([[[.1, .1, .4, .4], [.1, .1, .4, .4], [.5, .5, .9, .9]],
                          [[.1, .1, .4, .4], [.1, .1, .4, .4], [.5, .5, .9, .9]]])
    labels = torch.tensor([[0, 0, 1], [0, 1, 2]])
    scores = torch.tensor([[.8, .8, .6], [.8, .7, .6]])
    valid = torch.tensor([[True, True, True], [False, False, False]])
    return f, (boxes, labels, scores, valid)


def activate_readout(model):
    # Fixture weights, no optimization: prevent invariance tests passing only because output heads are zero.
    with torch.no_grad():
        model.classifier.weight.normal_(0., .03)
        model.box_head[-1].weight.normal_(0., .02)


def head_contracts():
    models = {arm: CompletionHead(arm).eval() for arm in ("N", "S", "R")}
    COUNTS["synthetic_head_builds"] += 3
    n, s, r = (models[a] for a in ("N", "S", "R"))
    copy_common(n, s)
    copy_common(n, r)
    assert parameter_count(s) - parameter_count(n) == 133888
    assert parameter_count(n) == parameter_count(r)
    f, candidates = fixtures()
    with torch.no_grad():
        initial = {a: head_call(model, f, candidates) for a, model in models.items()}
        for key in ("pred_boxes", "pred_logits", "query_states"):
            close(initial["S"][key], initial["N"][key], 2e-6)
        for arm, out in initial.items():
            assert bool((out["pred_logits"].sigmoid() < .001).all())
            close(out["pred_logits"].sigmoid(), torch.full_like(out["pred_logits"], 1e-4), 1e-10)
            expected = out["plan"]["residual"] if arm == "R" else out["mass"].detach()
            dv, db = density_inputs(expected)
            close(out["density_value"], dv, 1e-7)
            close(out["density_bias"], db, 1e-7)
        close(initial["S"]["counts"], torch.tensor([[2., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [0.]*10]))
        # Complete RGB -> 16-square features -> four boxes/logits path, still synthetic.
        rgb = torch.rand(1, 3, 128, 128)
        output = head_call(n, rgb, tuple(x[:1] for x in candidates), rgb=True)
        assert output["pred_boxes"].shape == (1, 4, 4)
        assert output["pred_logits"].shape == (1, 4, 10)
    activate_readout(n)
    copy_common(n, s)
    copy_common(n, r)
    with torch.no_grad():
        s.count_projection.weight.normal_(0., .05)
        for block in s.blocks:
            block.p_attention.out_proj.weight.normal_(0., .03)
    # Both S pathways are now nonzero. Tests must compare hidden states as well as nonconstant predictions.
    with torch.no_grad():
        for model in (s, r):
            baseline = head_call(model, f, candidates)
            order = torch.tensor([2, 0, 1])
            permuted = head_call(model, f, tuple(x[:, order] for x in candidates))
            for key in ("pred_boxes", "pred_logits", "query_states"):
                close(baseline[key], permuted[key], 2e-5)
            boxes, cls, score, valid = candidates
            padded = (torch.cat((boxes, torch.full((2, 2, 4), float("nan"))), 1),
                      torch.cat((cls, torch.full((2, 2), -99, dtype=torch.long)), 1),
                      torch.cat((score, torch.full((2, 2), float("nan"))), 1),
                      torch.cat((valid, torch.zeros((2, 2), dtype=torch.bool)), 1))
            with_padding = head_call(model, f, padded)
            for key in ("pred_boxes", "pred_logits", "query_states"):
                close(baseline[key], with_padding[key], 2e-5)
        n_base = head_call(n, f, candidates)
        changed = (candidates[0] + .2, (candidates[1] + 3) % 10,
                   candidates[2] * .5, ~candidates[3])
        n_changed = head_call(n, f, changed)
        for key in ("pred_boxes", "pred_logits", "query_states"):
            close(n_base[key], n_changed[key], 0.)
        empty = tuple(x[:, :0] for x in candidates)
        n_empty, s_empty = head_call(n, f, empty), head_call(s, f, empty)
        for key in ("pred_boxes", "pred_logits", "query_states"):
            close(n_empty[key], s_empty[key], 2e-6)
        # Explicit h contract, rather than a vacuous claim that normalized attention counts duplicates.
        h_one = torch.zeros(1, CLASSES)
        h_two = torch.zeros(1, CLASSES)
        h_one[:, 0], h_two[:, 0] = math_log(2.), math_log(3.)
        identical = torch.randn(1, DIM)
        close(torch.zeros(1).softmax(0) @ identical,
              torch.zeros(2).softmax(0) @ identical.expand(2, -1), 0.)
        assert not torch.equal(s.count_projection(h_one), s.count_projection(h_two))
    return models, dict(S_extra_parameters=133888, S_init_N_hidden_and_output_parity=True,
                        unified_density_interfaces=True, class_prior_below_native_threshold=True,
                        native_full_fusion_not_implemented=True, synthetic_RGB_forward=True,
                        nonzero_S_permutation_padding_invariance=True, R_permutation_padding_invariance=True,
                        N_forward_P_independence=True, empty_P_S_equals_N=True,
                        log1p_class_count_distinguishes_multiplicity=True)


def math_log(value):
    # Python scalar only; no data-dependent calibration.
    import math
    return math.log(value)


def detached_mass_shared_feature_contracts(models):
    _, candidates = fixtures()
    small = tuple(x[:1] for x in candidates)
    report = {}
    for arm in ("N", "S", "R"):
        model = models[arm]
        out = head_call(model, torch.rand(1, 3, 128, 128), small, rgb=True)
        loss = out["pred_boxes"][..., 0].sum() + out["pred_logits"].square().mean()
        mass_w, mass_b, shared_w, feature = grads(loss, (model.mass_head.weight, model.mass_head.bias,
                                                       model.trunk.down[0].weight, out["features"]))
        assert mass_w is None and mass_b is None
        assert shared_w is not None and bool(torch.isfinite(shared_w).all()) and shared_w.abs().max().item() > 0
        assert feature is not None and feature.abs().max().item() > 0
        # Density loss has its own direct mass-head path even though detection cannot take that shortcut.
        fresh = head_call(model, torch.rand(1, 3, 128, 128), small, rgb=True)
        density_grads = grads(fresh["mass"].sum(), (model.mass_head.weight, model.mass_head.bias))
        assert density_grads[1] is not None and density_grads[1].abs().max().item() > 0
        report[arm] = dict(detection_mass_head_grad_none=True,
                           detection_shared_trunk_grad_nonzero=True,
                           detection_feature_grad_nonzero=True, density_head_direct_grad=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "synthetic_contracts.json")
    args = parser.parse_args()
    assert not args.output.exists(), "Refuse to overwrite previous synthetic evidence"
    start = time.perf_counter()
    sources = {str(p): sha(p) for p in (HERE / "core.py", pathlib.Path(__file__).resolve())}
    result = dict(state="running", scope="synthetic contracts only; no learnability/AP claim",
                  sources=sources, synthetic_parameter_fixture_writes=True)
    try:
        result["resources"] = resources()
        result["operator"] = operator_contracts()
        result["hand_and_gradient"] = hand_and_gradient_contracts()
        assert psutil.virtual_memory().available / 2**30 > 8
        models, result["heads"] = head_contracts()
        result["gradient_routes"] = detached_mass_shared_feature_contracts(models)
        result["state"] = "complete"
    except Exception:
        result["state"] = "FAILED_IMPLEMENTATION"
        result["traceback"] = traceback.format_exc()
    result["counts"] = COUNTS
    result["seconds"] = time.perf_counter() - start
    result["sources_unchanged"] = {p: sha(pathlib.Path(p)) == h for p, h in sources.items()}
    if not all(result["sources_unchanged"].values()):
        result["state"] = "FAILED_IMPLEMENTATION"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result["state"], counts=COUNTS, seconds=result["seconds"],
                          output=str(args.output)), ensure_ascii=False))
    if result["state"] != "complete":
        print(result.get("traceback", "Source hash mismatch"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
