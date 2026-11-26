"""Plot existing verified observations; never load or execute a neural model."""
from datetime import datetime, timezone
from hashlib import sha256
import csv
import io
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
REMOTE = PAPER / '远程公平对照_20261103'
RESUME = REMOTE / '断点续训_20261105'
sys.path.insert(0, str(HERE / 'plot_runtime'))
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
os.environ['MPLCONFIGDIR'] = str(HERE / 'plot_cache')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def exclusive_json(path, data):
    if path.exists():
        assert read(path) == data, 'An existing data artifact differs; refusing overwrite'
        return
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    pair_path = RESUME / 'BASE_FULL_pair_verified_20261108.json'
    pair = read(pair_path)
    assert pair['status'] == 'RESUMED_PAIR_ENDPOINTS_VERIFIED_PENDING_MANUSCRIPT_REVIEW'
    index = read(HERE.parent / '主要结论证据索引/主要结论证据索引_第三十二轮.json')
    assert digest(pair_path) == index['evidence_files'][pair_path.relative_to(PAPER).as_posix()]
    paths = {
        'BASE_original': REMOTE / 'remote_evidence/interrupted_BASE_20261105/runs/BASE/epochs.jsonl',
        'BASE_to308': RESUME / 'backup_epoch0307/epochs.jsonl',
        'BASE_to320': RESUME / 'BASE_30166_r308/terminal_evidence/run/epochs.jsonl',
        'FULL_original': REMOTE / 'secondary_evidence/evidence_audits/FULL_best_epoch0000/metrics.json',
        'FULL_to320': RESUME / 'FULL_30848_20261106/backups/epochs/epoch0319/epochs.jsonl',
    }
    assert pair['lineages']['BASE']['resume_boundaries'] == [214, 308]
    assert digest(paths['BASE_to308']) == '0b23b0776689b2d22107e8a939e62222976b9b83885de6b45d3feaaa0f3f0c98'
    original = rows(paths['BASE_original'])
    middle = rows(paths['BASE_to308'])
    ending = rows(paths['BASE_to320'])
    assert len(middle) == 94 and len(ending) == 12
    full_anchor = read(paths['FULL_original'])
    assert full_anchor.pop('checkpoint_sha256') == pair['lineages']['FULL']['origin_sha256']
    streams = {'BASE': original[:214] + middle + ending,
               'FULL': [full_anchor] + rows(paths['FULL_to320'])}
    metric_names = list(pair['comparisons']['terminal']['absolute_percent']['BASE'])
    for arm, stream in streams.items():
        assert len(stream) == 320
        for epoch, item in enumerate(stream):
            assert item['epoch'] == epoch
            assert item['optimizer_step_attempts'] == (epoch + 1) * 1617
            assert item['protocol_sha256'] == pair['shared_conditions']['protocol_sha256']
            assert item['weight_source'] == 'ema.module'
            assert set(item['metrics']) == set(metric_names)
            assert all(math.isfinite(value) and 0 <= value <= 1 for value in item['metrics'].values())
        best = max(range(320), key=lambda i: (stream[i]['metrics']['AP'], -i))
        for role, epoch in [('terminal', 319), ('best', best)]:
            comparison = pair['comparisons'][role]
            assert comparison['epoch_indices'][arm] == epoch
            for name, value in comparison['absolute_percent'][arm].items():
                assert abs(stream[epoch]['metrics'][name] * 100 - value) <= 1e-12
    exclusive_json(HERE / 'paired_learning_curve_rows.json', {
        'scope': 'Observed joined continuations; all 320 completed rounds, unsmoothed. The 11 original BASE rows after the round-214 anchor are excluded, as in the verified endpoint audit.',
        'BASE': streams['BASE'], 'FULL': streams['FULL']})
    csv_path = HERE / 'paired_learning_curve_metrics.csv'
    csv_stream = io.StringIO(newline='')
    writer = csv.writer(csv_stream)
    writer.writerow(['arm', 'completed_round'] + metric_names)
    for arm, stream_rows in streams.items():
        for item in stream_rows:
            writer.writerow([arm, item['epoch'] + 1] + [item['metrics'][name] for name in metric_names])
    csv_bytes = csv_stream.getvalue().encode('utf-8')
    if csv_path.exists():
        assert csv_path.read_bytes() == csv_bytes, 'Existing plot data differ'
    else:
        csv_path.write_bytes(csv_bytes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), layout=None)
    colors = {'BASE': '#0072B2', 'FULL': '#D55E00'}
    for axis, metric, title, upper in zip(axes, ['AP', 'APs'],
                                         ['Validation AP', 'Small-object validation AP'], [25, 18]):
        axis.axvspan(296.5, 320.5, color='#E9E9E9', zorder=0)
        axis.axvline(296.5, color='#777777', linewidth=0.7, linestyle=':', zorder=1)
        for arm, stream in streams.items():
            axis.plot(range(1, 321), [100 * row['metrics'][metric] for row in stream],
                      color=colors[arm], linewidth=1.0,
                      linestyle='--' if arm == 'BASE' else '-', label=arm, zorder=2)
            best = pair['comparisons']['best']['epoch_indices'][arm]
            axis.scatter([best + 1], [100 * stream[best]['metrics'][metric]],
                         marker='*', s=60, color=colors[arm], edgecolors='black',
                         linewidths=0.35, zorder=4)
            axis.scatter([320], [100 * stream[319]['metrics'][metric]],
                         marker='o', s=22, color=colors[arm], edgecolors='black',
                         linewidths=0.35, zorder=5)
        axis.set(xlim=(0, 326), ylim=(0, upper), title=title,
                 xlabel='Completed training round', ylabel=f'{metric} (%)')
        axis.set_xticks([0, 80, 160, 240, 320])
        axis.grid(axis='y', color='#D6D6D6', linewidth=0.5, zorder=0)
    handles = [Line2D([0], [0], color=colors['BASE'], linestyle='--', linewidth=1.2, label='BASE'),
               Line2D([0], [0], color=colors['FULL'], linewidth=1.2, label='FULL'),
               Line2D([0], [0], color='white', marker='o', markerfacecolor='#777777',
                      markeredgecolor='black', markersize=4, label='Round 320'),
               Line2D([0], [0], color='white', marker='*', markerfacecolor='#777777',
                      markeredgecolor='black', markersize=8, label='Best-AP round')]
    figure.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, 0.005), fontsize=8, columnspacing=1.6, handlelength=2)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.24, top=0.87, wspace=0.28)
    outputs = []
    for suffix in ['png', 'pdf', 'svg']:
        output = HERE / f'paired_validation_curves.{suffix}'
        assert not output.exists()
        figure.savefig(output, dpi=300, facecolor='white', metadata={'Creator': 'Matplotlib'})
        outputs.append(output)
    plt.close(figure)
    exclusive_json(HERE / 'learning_curve_provenance.json', {
        'status': 'ALL_320_OBSERVED_ROUNDS_BOUND_TO_VERIFIED_PAIR',
        'observed_local_utc': datetime.now(timezone.utc).isoformat(),
        'pair_report_sha256': digest(pair_path),
        'source_sha256': {p.relative_to(PAPER).as_posix(): digest(p) for p in paths.values()},
        'source_slices': {'BASE': ['original[:214]', 'saved rounds 215–308', 'completed recovery rounds 309–320'],
                          'FULL': ['original round 1', 'completed continuation rounds 2–320']},
        'plotted_metrics': ['AP', 'APs'], 'plot_units': 'percent', 'smoothing': None,
        'endpoint_rules': 'Circles: actual round 320. Stars: best-AP round; APs follows that same AP-selected checkpoint.',
        'shaded_rounds': [297, 320], 'shaded_meaning': 'Validation-dependent restoration stage',
        'BASE_resume_boundaries': [214, 308], 'FULL_resume_boundaries': [1],
        'raw_csv_sha256': digest(csv_path),
        'figure_sha256': {p.name: digest(p) for p in outputs},
        'plot_runtime': {'python': sys.version, 'matplotlib': matplotlib.__version__},
        'new_model_execution': False, 'new_evaluation': False})
    print(json.dumps({'status': 'ALL_320_OBSERVED_ROUNDS_BOUND_TO_VERIFIED_PAIR',
                      'figure': str(outputs[0]), 'rounds_per_arm': 320}))


if __name__ == '__main__':
    main()
