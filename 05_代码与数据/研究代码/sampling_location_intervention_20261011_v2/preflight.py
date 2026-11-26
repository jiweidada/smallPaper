"""Read-only runtime preflight: parse configuration, never construct a model.

The only created artifact is --output, a fresh JSON file. Existing target-policy
tools, bundles, native sources, checkpoints and evidence are read and verified.
No detector, suffix, evaluator, optimizer, CUDA or training call is permitted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True

EXPECTED_DEPENDENCIES = {
    "verify_candidate_reload.py": "72552be94aaaef06960c12e23c8529c8247353f07a1348539b2a461ecc4aeccf",
    "closeout_protocol.json": "80b93619946243446602ef64809aacd62d3ec4f05b5f6f9aefe4b40aac1777b8",
}
EXPECTED_RUNTIME_SHA = "d373ca671abdc2ae4cf0123a9dd52338aa5ac98cad8ed3dce44c1063b8f92d01"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def module_location(module, path):
    actual = Path(module.__file__).resolve()
    assert actual == Path(path).resolve(), (module.__name__, str(actual), str(path))
    return dict(path=str(actual), sha256=sha(actual))


def gpu_snapshot():
    uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True, timeout=30).strip()
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True, timeout=30).strip()
    assert not raw, ("GPU compute applications are active", raw)
    return dict(utc=utc(), uuid=uuid, compute_application_pids=[])


def check_signature(callable_object, args=(), kwargs=None):
    signature = inspect.signature(callable_object)
    signature.bind(*args, **(kwargs or {}))
    return str(signature)


def run(args):
    protocol_path = Path(args.protocol).resolve()
    output_path = Path(args.output).resolve()
    assert not output_path.exists(), "Preflight output must be new; never overwrite evidence"
    assert output_path.parent.is_dir(), "Create output parent before preflight"
    counts = dict(model_constructions=0, detector_forwards=0, decoder_suffix_calls=0,
                  BASE_forwards=0, evaluator_calls=0, optimizer_updates=0,
                  epochs_added=0, CUDA_initializations=0, forbidden_module_calls=0,
                  forbidden_construction_attempts=0, forbidden_CUDA_attempts=0,
                  configuration_objects_parsed=0)
    report = dict(status="READ_ONLY_PREFLIGHT_STARTED", seed=args.seed,
                  protocol_path=str(protocol_path), protocol_sha256=sha(protocol_path),
                  preflight_path=str(Path(__file__).resolve()), preflight_sha256=sha(__file__),
                  started_utc=utc(), counts=counts, paths={}, imported_modules={}, checks_completed=[])
    guards = []
    try:
        p = read(protocol_path)
        assert p["schema"] == "sampling_location_suffix_intervention_v2"
        root = protocol_path.parent
        assert Path(__file__).resolve().parent == root
        assert {"runner.py", "controller.py", "preflight.py"} <= set(p["code_sha256"])
        for name, expected in p["code_sha256"].items():
            assert Path(name).name == name and (root / name).is_file(), name
            assert sha(root / name) == expected, ("Frozen execution source", name)
        assert p["dependency_sha256"] == EXPECTED_DEPENDENCIES
        report["code_sha256"] = p["code_sha256"]
        report["checks_completed"].append("frozen_execution_sources")

        spec = p["seeds"][str(args.seed)]
        tools = Path(spec["closeout_tools"]).resolve()
        bundle = Path(spec["bundle"]).resolve()
        assert tools.name == "closeout_tools_v1" and tools.is_dir()
        assert bundle.name == "bundle" and bundle.is_dir()
        assert bundle.parent.parent == tools.parent
        assert bundle.parent.name == f"seed{args.seed}_target_protect"
        report["paths"].update(closeout_tools=str(tools), bundle=str(bundle))
        dependency_hashes = {}
        for name, expected in EXPECTED_DEPENDENCIES.items():
            path = tools / name
            assert path.is_file(), ("Missing dependency", str(path))
            dependency_hashes[name] = sha(path)
            assert dependency_hashes[name] == expected, name
        report["dependency_sha256"] = dependency_hashes

        # A fresh interpreter is required so no different module can satisfy an
        # import merely because it was already cached in sys.modules.
        assert not any(name in sys.modules for name in
                       ("verify_candidate_reload", "runtime", "engine", "evaluator_binding"))
        sys.path.insert(0, str(tools))
        importlib.invalidate_caches()
        reload_tools = importlib.import_module("verify_candidate_reload")
        report["imported_modules"]["verify_candidate_reload"] = module_location(
            reload_tools, tools / "verify_candidate_reload.py")
        api = {
            "verify_tools": check_signature(reload_tools.verify_tools),
            "bound_runtime": check_signature(reload_tools.bound_runtime, (bundle,)),
            "assert_trainers_stopped": check_signature(reload_tools.assert_trainers_stopped, (bundle.parent,)),
        }
        closeout = reload_tools.verify_tools()
        assert closeout["schema"] == "scale_protection12_closeout_v1"
        report["verified_closeout_files"] = closeout["files"]
        report["checks_completed"].append("actual_closeout_import_and_verify_tools")

        code, receipt, plan, paths = reload_tools.bound_runtime(bundle)
        assert code.resolve() == tools.parent / "code_v2"
        assert plan["seed"] == args.seed and plan["arm"] == "target_protect"
        assert receipt["identity"]["seed"] == args.seed and receipt["identity"]["arm"] == "target_protect"
        assert paths["screen_root"].resolve() == bundle.parent
        assert paths["experiment_root"].resolve() == tools.parent
        reload_tools.assert_trainers_stopped(paths["screen_root"])
        runtime = importlib.import_module("runtime")
        binding = importlib.import_module("evaluator_binding")
        report["imported_modules"]["runtime"] = module_location(runtime, code / "runtime.py")
        report["imported_modules"]["evaluator_binding"] = module_location(binding, code / "evaluator_binding.py")
        assert sha(code / "runtime.py") == EXPECTED_RUNTIME_SHA
        report["paths"].update({key: str(value.resolve()) for key, value in paths.items()})
        report["paths"]["code"] = str(code.resolve())
        report["bundle_receipt_sha256"] = sha(bundle / "receipt.json")
        report["bundle_plan_sha256"] = sha(bundle / "plan.json")
        report["bundle_resolved_sha256"] = sha(bundle / "resolved.yml")
        source_binding = read(code / "source_binding.json")
        report["native_source_count"] = len(source_binding["native_files_sha256"])
        report["native_source_binding_sha256"] = sha(code / "source_binding.json")
        report["native_sources_and_evaluator_files_verified_by_bound_runtime"] = True
        report["implementation_sha256"] = receipt["identity"]["implementation"]
        report["checks_completed"].append("actual_bound_runtime_and_trainers_stopped")

        training = read(paths["output"] / "training_completion.json")
        assert training["status"] == "NATIVE_FIT_FINISHED_EXACT12"
        assert training["completed_epochs"] == 12 and training["entered_epochs"] == list(range(12))
        assert training["identity"] == receipt["identity"]
        assert read(paths["screen_root"] / "controller_completion.json")["status"] == "COMPLETE_FIXED12"
        assert read(paths["screen_root"] / "candidate_completion.json")["exit_code"] == 0
        checkpoint = paths["output"] / "last.pth"
        assert checkpoint.is_file()
        checkpoint_hash = sha(checkpoint)
        assert checkpoint_hash == spec["checkpoint_sha256"] == training["last_checkpoint_sha256"]
        report["checkpoint"] = dict(path=str(checkpoint.resolve()), bytes=checkpoint.stat().st_size,
                                    sha256=checkpoint_hash, deserialized=False)
        gpu_before = gpu_snapshot()
        assert gpu_before["uuid"] == spec["gpu_uuid"]
        assert gpu_before["uuid"] == read(code / "training_protocol.json")["GPU_uuid"][str(args.seed)]
        report["gpu_before"] = gpu_before
        report["checks_completed"].append("completed_fixed12_checkpoint_hash_and_idle_GPU")

        # Block CUDA initialization and nn.Module forward dispatch before native
        # imports. Merely importing torch is allowed; no tensors are constructed
        # by this preflight and no checkpoint is deserialized.
        import torch
        assert not torch.cuda.is_initialized()

        def patch(obj, name, value):
            original = getattr(obj, name)
            guards.append((obj, name, original))
            setattr(obj, name, value)

        def forbid_cuda(*unused_args, **unused_kwargs):
            counts["forbidden_CUDA_attempts"] += 1
            raise AssertionError("CUDA execution is forbidden during read-only preflight")

        def forbid_module_call(*unused_args, **unused_kwargs):
            counts["forbidden_module_calls"] += 1
            raise AssertionError("nn.Module execution is forbidden during read-only preflight")

        def forbid_construction(*unused_args, **unused_kwargs):
            counts["forbidden_construction_attempts"] += 1
            raise AssertionError("Config-driven model/component construction is forbidden during preflight")

        patch(torch.cuda, "_lazy_init", forbid_cuda)
        patch(torch.cuda, "init", forbid_cuda)
        patch(torch.nn.Module, "cuda", forbid_cuda)
        patch(torch.Tensor, "cuda", forbid_cuda)
        patch(torch.nn.Module, "_call_impl", forbid_module_call)
        api["runtime.activate"] = check_signature(runtime.activate, (paths["native_root"],))
        api["runtime.resolved_config"] = check_signature(runtime.resolved_config, (plan, paths))
        api["runtime.register_model"] = check_signature(runtime.register_model)
        runtime.activate(paths["native_root"])
        binding.activate(paths["evaluator_root"], receipt["identity"]["evaluator_binding"])
        runtime.register_model()  # Class registration only; never instantiation.
        core = importlib.import_module("engine.core")
        yaml_module = importlib.import_module("engine.core.yaml_config")
        workspace = importlib.import_module("engine.core.workspace")
        native_root = paths["native_root"]
        report["imported_modules"]["engine.core.yaml_config"] = module_location(
            yaml_module, native_root / "engine/core/yaml_config.py")
        report["imported_modules"]["engine.core.workspace"] = module_location(
            workspace, native_root / "engine/core/workspace.py")
        patch(yaml_module, "create", forbid_construction)
        patch(workspace, "create", forbid_construction)
        patch(core, "create", forbid_construction)
        patch(core.YAMLConfig, "model", property(forbid_construction))

        # Parse the already saved JSON-compatible YAML through its real API.
        # resolved_config verifies the historical recipe first. The second parse
        # only overrides output_dir exactly as the runner's resolved_config does.
        original_config = runtime.resolved_config(plan, paths)
        assert read(bundle / "resolved.yml") == original_config
        cfg = core.YAMLConfig(str(bundle / "resolved.yml"))
        counts["configuration_objects_parsed"] += 1
        assert cfg.yaml_cfg == original_config, "Actual YAMLConfig parse changed resolved configuration values"
        desired_output = root / f"seed{args.seed}" / "outputs"
        runner_config = runtime.resolved_config(plan, dict(paths, output=desired_output))
        assert {key for key in runner_config if runner_config[key] != original_config[key]} <= {"output_dir"}
        runner_cfg = core.YAMLConfig(str(bundle / "resolved.yml"), output_dir=str(desired_output))
        counts["configuration_objects_parsed"] += 1
        assert runner_cfg.yaml_cfg == runner_config
        lazy_fields = ("_model", "_criterion", "_postprocessor", "_optimizer", "_train_dataloader",
                       "_val_dataloader", "_evaluator", "_ema", "_scaler", "_writer")
        for instance in (cfg, runner_cfg):
            assert all(instance.__dict__[name] is None for name in lazy_fields)
        assert "ResidualGeometryDEIM" in workspace.GLOBAL_CONFIG
        report["configuration"] = dict(
            source_path=str((bundle / "resolved.yml").resolve()),
            parsed_source_exactly_matches_resolved_config=True,
            runner_output_dir=str(desired_output), runner_output_override_parse_exact=True,
            parsed_model_name=cfg.yaml_cfg["model"], registered_without_instantiation=True,
            lazy_fields_verified_unconstructed=list(lazy_fields),
            resolved_config_sha256=hashlib.sha256(json.dumps(original_config, sort_keys=True, ensure_ascii=False,
                                                            allow_nan=False).encode("utf8")).hexdigest())

        # Resolve the runner's remaining named imports and signatures without
        # invoking attention backends, tensor functions or distributed setup.
        module_specs = {
            "engine.misc.dist_utils": ("setup_distributed", "cleanup"),
            "engine.misc.flash_attention": ("FlashSelfAttention",),
            "engine.deim.dfine_utils": ("distance2bbox", "weighting_function"),
            "engine.deim.utils": ("inverse_sigmoid",),
            "engine.misc.deformable_detr_msda": ("should_use_deformable_detr_msda", "deformable_detr_msda_core_default"),
        }
        for module_name, symbols in module_specs.items():
            module = importlib.import_module(module_name)
            expected_path = native_root / (module_name.replace(".", "/") + ".py")
            report["imported_modules"][module_name] = module_location(module, expected_path)
            for name in symbols:
                value = getattr(module, name)
                assert callable(value), (module_name, name)
                api[module_name + "." + name] = str(inspect.signature(value))
        dist = sys.modules["engine.misc.dist_utils"]
        check_signature(dist.setup_distributed, kwargs=dict(print_rank=0, print_method="builtin", seed=args.seed))
        msda = sys.modules["engine.misc.deformable_detr_msda"]
        check_signature(msda.should_use_deformable_detr_msda, (None, None, None))
        check_signature(msda.deformable_detr_msda_core_default, (None, None, None, None, None))
        dfine_utils = sys.modules["engine.deim.dfine_utils"]
        check_signature(dfine_utils.distance2bbox, (None, None, None))
        check_signature(dfine_utils.weighting_function, (None, None, None))
        report["verified_API_signatures"] = api
        assert not torch.cuda.is_initialized()
        assert all(counts[name] == 0 for name in counts if name != "configuration_objects_parsed")
        report["torch_version"] = torch.__version__
        report["cuda_initialized"] = False
        report["checks_completed"].append("actual_native_imports_YAMLConfig_parse_and_API_signatures_without_construction")
        report["gpu_after"] = gpu_snapshot()
        assert report["gpu_after"]["uuid"] == spec["gpu_uuid"]
        report["status"] = "PASS_READ_ONLY_PREFLIGHT_NO_MODEL"
        report["completed_utc"] = utc()
        report["scope"] = "Read-only dependency, binding, lifecycle, checkpoint hash, native imports and configuration parse. No numerical suffix validity or detection efficacy has been tested."
        write(output_path, report)
        print(json.dumps(dict(status=report["status"], seed=args.seed,
                              protocol_sha256=report["protocol_sha256"], output=str(output_path), counts=counts)), flush=True)
    except BaseException:
        report["status"] = "FAIL_READ_ONLY_PREFLIGHT_NO_MODEL"
        report["traceback"] = traceback.format_exc()
        report["completed_utc"] = utc()
        report["automatic_execution_after_failure"] = False
        write(output_path, report)
        raise
    finally:
        for obj, name, value in reversed(guards):
            setattr(obj, name, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
