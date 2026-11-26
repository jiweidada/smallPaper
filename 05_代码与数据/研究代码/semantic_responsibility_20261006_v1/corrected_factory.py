"""Explicit configuration binding for an isolated research baseline.

The historical snapshot is imported read-only and hash-checked. No shared file is
patched. This module repairs configuration transport; it is not an innovation.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORKSPACE = next(p for p in HERE.parents if (p / 'agent.md').exists())
SNAPSHOT = HERE.parent / 'local_candidates_20261004_v2'
RECEIPT = WORKSPACE / '英文小论文/data/00_实验计划与证据索引/模型数据与既有文献联合调研_20261006_v1/03d_匹配配置差异与冻结指派结果.json'


def source_hashes():
    return {str(p.relative_to(SNAPSHOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SNAPSHOT.rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.yml'}}


def verify_source():
    expected = json.loads(RECEIPT.read_text(encoding='utf-8'))['binding']['source_hashes']
    actual = source_hashes()
    if actual != expected:
        raise RuntimeError('Historical source hash set changed; stop instead of importing a changed baseline.')
    return actual


verify_source()
sys.path.insert(0, str(SNAPSHOT))
import bootstrap  # noqa: E402
from factory import native_recipe  # noqa: E402
from engine.deim.deim_criterion import DEIMCriterion  # noqa: E402
from engine.deim.matcher import HungarianMatcher  # noqa: E402


def build_corrected_criterion(recipe=None):
    """Bind the declared global flag; fail on ambiguous/conflicting declarations."""
    recipe = copy.deepcopy(native_recipe() if recipe is None else recipe)
    if type(recipe.get('use_focal_loss')) is not bool:
        raise ValueError('The isolated baseline requires an explicit boolean use_focal_loss.')
    kwargs = recipe['DEIMCriterion']
    matcher = kwargs.pop('matcher')
    if matcher.pop('type') != 'HungarianMatcher':
        raise ValueError('Unexpected matcher type.')
    declared = recipe['use_focal_loss']
    if 'use_focal_loss' in matcher and matcher['use_focal_loss'] != declared:
        raise ValueError('Conflicting global and matcher-local use_focal_loss values.')
    matcher['use_focal_loss'] = declared
    return DEIMCriterion(matcher=HungarianMatcher(**matcher), num_classes=10, **kwargs)
