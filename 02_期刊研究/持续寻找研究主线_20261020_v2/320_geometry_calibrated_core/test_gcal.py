"""Parent-run synthetic G contracts only; never load real inputs/checkpoints.

Run python -s -X utf8 -B test_gcal.py. A result file is created exclusively.
Synthetic fixture parameter writes are not optimization or evidence of learning.
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
assert sys.flags.no_user_site, "Run python -s"
assert RUNTIME.is_dir(), RUNTIME
sys.path[:0] = [str(HERE), str(RUNTIME)]
import psutil
import torch
from gcal import GeometryCalibratedHead, copy_R_common, base, FROZEN_CORE, FROZEN_SHA256

COUNTS = dict(synthetic_head_builds=0, synthetic_head_forwards=0,
              synthetic_autograd_calls=0, optimizer_steps=0, real_image_reads=0,
              checkpoint_loads=0, native_imports=0, global_detector_forwards=0,
              heldout_reads=0)


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
    torch.manual_seed(2026102105)
    return dict(threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
                affinity=proc.cpu_affinity(), priority="BelowNormal",
                free_RAM_GiB=psutil.virtual_memory().available / 2**30,
                torch=torch.__version__, torch_file=torch.__file__, executable=sys.executable,
                no_user_site=bool(sys.flags.no_user_site), mha_fastpath=False,
                MKL_THREADING_LAYER=os.environ["MKL_THREADING_LAYER"])


def close(a, b, atol=1e-7):
    torch.testing.assert_close(a, b, atol=atol, rtol=0.)


def call(model, data, candidates, rgb=False):
    COUNTS["synthetic_head_forwards"] += 1
    return model(data, *candidates) if rgb else model.forward_features(data, *candidates)


def grads(loss, inputs):
    COUNTS["synthetic_autograd_calls"] += 1
    return torch.autograd.grad(loss, inputs, allow_unused=True)


def fixture():
    # One candidate in each of two classes; row two is empty padding.
    f = torch.randn(2, base.DIM, base.SIDE, base.SIDE)
    boxes = torch.tensor([[[-.1, -.1, 1.1, 1.1], [.1, .1, .9, .9]],
                          [[-.1, -.1, 1.1, 1.1], [.1, .1, .9, .9]]])
    classes = torch.tensor([[0, 1], [0, 1]])
    scores = torch.tensor([[.8, .7], [.8, .7]])
    valid = torch.tensor([[True, True], [False, False]])
    return f, (boxes, classes, scores, valid)


def plan_contract(plan, valid):
    close(plan["T"].sum(1) + plan["z"], valid.float(), 3e-7)
    assert bool((plan["explained"] <= plan["mass"] + 1e-7).all())
    close(plan["residual"], (plan["mass"] - plan["u"]).clamp_min(0.), 1e-7)
    assert all(bool(torch.isfinite(v).all()) for v in plan.values())


def initial_and_routing_contracts():
    r, g = base.CompletionHead("R").eval(), GeometryCalibratedHead().eval()
    COUNTS["synthetic_head_builds"] += 2
    copy_R_common(r, g)
    assert base.parameter_count(g) - base.parameter_count(r) == 10
    close(g.class_null, torch.zeros_like(g.class_null), 0.)
    f, candidates = fixture()
    with torch.no_grad():
        ro, go = call(r, f, candidates), call(g, f, candidates)
        assert go["plan"] is go["aux_plan"]
        for key in ro["plan"]:
            close(ro["plan"][key], go["aux_plan"][key])
            close(ro["plan"][key], go["generation_plan"][key])
        for key in ("density_field", "density_value", "density_bias", "query_states",
                    "pred_boxes", "pred_logits"):
            close(ro[key], go[key], 2e-6)
        assert go["query_states"].abs().max().item() > 0
        assert (go["mass"] - go["density_field"]).max().item() > 0
        assert bool((go["pred_logits"].sigmoid() < .001).all())
        plan_contract(go["generation_plan"], candidates[-1])
        # Nonzero readout fixture prevents later output invariance from being vacuous.
        r.classifier.weight.normal_(0., .03)
        r.box_head[-1].weight.normal_(0., .02)
        copy_R_common(r, g)
        ro, baseline = call(r, f, candidates), call(g, f, candidates)
        for key in ("query_states", "pred_boxes", "pred_logits"):
            close(ro[key], baseline[key], 2e-6)
        # Only the learned auxiliary C/theta and kappa/theta are altered.
        g.association.query.weight.normal_(0., .08)
        g.association.query.bias.fill_(.1)
        g.association.null[-1].weight.normal_(0., .05)
        g.association.null[-1].bias.fill_(.7)
        changed_aux = call(g, f, candidates)
        assert not torch.allclose(baseline["aux_plan"]["T"], changed_aux["aux_plan"]["T"])
        for key in ("density_field", "density_value", "density_bias", "query_states",
                    "pred_boxes", "pred_logits"):
            close(baseline[key], changed_aux[key], 0.)
        g.class_null[0] = .8
        changed_class = call(g, f, candidates)
        assert not torch.equal(changed_class["density_field"][:, 0], changed_aux["density_field"][:, 0])
        close(changed_class["density_field"][:, 1:], changed_aux["density_field"][:, 1:], 0.)
        for key in changed_aux["aux_plan"]:
            close(changed_class["aux_plan"][key], changed_aux["aux_plan"][key], 0.)
        g.class_null.zero_()
        empty = tuple(x[:, :0] for x in candidates)
        re, ge = call(r, f, empty), call(g, f, empty)
        for key in ("density_field", "density_value", "density_bias", "query_states",
                    "pred_boxes", "pred_logits"):
            close(re[key], ge[key], 2e-6)
        close(ge["density_field"], ge["mass"], 0.)
        assert ge["generation_plan"]["T"].shape == (2, 256, 0)
    return g, dict(extra_parameters=10, common_weight_initial_plan_and_hidden_parity=True,
                   common_weight_nonzero_readout_parity=True, initial_score_below_threshold=True,
                   learned_aux_change_does_not_change_G_generation=True,
                   class_null_changes_only_corresponding_residual_class=True,
                   class_null_does_not_change_aux_plan=True,
                   empty_P_safe_and_R_parity=True, column_row_clip_contracts=True)


def gradient_contracts(g):
    _, candidates = fixture()
    small = tuple(x[:1] for x in candidates)
    out = call(g, torch.rand(1, 3, 128, 128), small, rgb=True)
    assert bool((out["generation_plan"]["u"][:, :2] < out["mass"].detach()[:, :2]).all())
    # A synthetic detection readout objective, not the proposed native/task loss.
    loss = out["pred_boxes"][..., 0].sum() + out["pred_logits"].square().mean()
    inputs = (g.class_null, g.mass_head.weight, g.mass_head.bias,
              g.trunk.down[0].weight, out["features"],
              g.association.query.weight, g.association.null[-1].weight)
    k, mw, mb, shared, feature, aq, an = grads(loss, inputs)
    assert k is not None and bool(torch.isfinite(k).all()) and k[:2].abs().max().item() > 0
    close(k[2:], torch.zeros_like(k[2:]), 0.)
    assert mw is None and mb is None
    assert shared is not None and bool(torch.isfinite(shared).all()) and shared.abs().max().item() > 0
    assert feature is not None and bool(torch.isfinite(feature).all()) and feature.abs().max().item() > 0
    assert aq is None and an is None
    # A separate strict-overload scalar fixture; no derivative claim at the equality kink.
    m = torch.zeros(1, base.CLASSES, 1, dtype=torch.float64)
    m[:, 0] = .1
    kappa = torch.zeros(base.CLASSES, dtype=torch.float64, requires_grad=True)
    cost = torch.zeros(1, 1, 1, dtype=torch.float64)
    cls, valid = torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        kappa[0] = 4.
    plan = base.independent_cap(m, cost, kappa[cls], cls, valid)
    assert plan["u"][0, 0, 0].item() > .1
    close(plan["T"], torch.full_like(plan["T"], .1), 1e-12)
    close(plan["z"], torch.full_like(plan["z"], .9), 1e-12)
    (kg,) = grads(plan["residual"].sum() - plan["T"].log().sum() - plan["z"].log().sum(), (kappa,))
    close(kg, torch.zeros_like(kg), 1e-12)
    return dict(detection_class_null_grad_nonzero=True, class_null_gradient=k.detach().tolist(),
                detection_fixture_affected_rows_strictly_unsaturated=True,
                detection_mass_head_grad_none=True, detection_shared_trunk_grad_nonzero=True,
                detection_feature_grad_nonzero=True, detection_learned_aux_grad_none=True,
                strict_overload_zero_kappa_gradient=True, equality_subgradient_not_assumed=True,
                no_optimizer_or_auxiliary_task_loss=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=HERE / "synthetic_contracts.json")
    args = parser.parse_args()
    assert not args.output.exists(), "Refuse to overwrite previous evidence"
    sources = {str(p): sha(p) for p in (HERE / "gcal.py", pathlib.Path(__file__).resolve(),
                                      FROZEN_CORE, FROZEN_CORE.with_name("test_core.py"))}
    assert sources[str(FROZEN_CORE)] == FROZEN_SHA256
    assert sources[str(FROZEN_CORE.with_name("test_core.py"))] == "824da320119bb66b96da7a8fc0aa436904c0c9ee7b2b7fded0faca57c8c3b13e"
    started = time.perf_counter()
    result = dict(state="running", scope="synthetic operator/routing contracts; no learnability or AP claim",
                  sources=sources, synthetic_fixture_parameter_writes=True,
                  inherited_311_tests_not_rerun=True,
                  class_null_optimizer_contract="det-only, no weight decay; no optimizer implemented")
    try:
        result["resources"] = resources()
        g, result["heads"] = initial_and_routing_contracts()
        assert psutil.virtual_memory().available / 2**30 > 8
        result["gradients"] = gradient_contracts(g)
        result["state"] = "complete"
    except Exception:
        result["state"] = "FAILED_IMPLEMENTATION"
        result["traceback"] = traceback.format_exc()
    result["counts"] = COUNTS
    result["seconds"] = time.perf_counter() - started
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
