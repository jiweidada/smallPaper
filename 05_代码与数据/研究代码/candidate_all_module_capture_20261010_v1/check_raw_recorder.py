"""Actual CPU engineering fixture; not a detection model or efficacy experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from raw_module_recorder import RawModuleRecorder, write


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Identity()
        self.relu = torch.nn.ReLU(inplace=True)
        self.unused = torch.nn.Linear(2, 2)

    def forward(self, x):
        before = self.first(x)
        self.relu(before)
        repeated = self.first(before)
        return {'nested': (repeated, [repeated.to(torch.bfloat16), None]), 'flag': True}


def main(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    model = Fixture().eval()
    original = torch.tensor([[-2., 3.], [1., -4.]])
    with torch.no_grad():
        expected = model(original.clone())
        recorder = RawModuleRecorder(model, destination / 'capture')
        try:
            x = original.clone()
            recorder.begin({'scope': 'CPU engineering fixture'}, x, [])
            output = model(x)
            record = recorder.finish()
        finally:
            recorder.close()
    assert torch.equal(output['nested'][0], expected['nested'][0])
    assert torch.equal(output['nested'][1][0], expected['nested'][1][0])
    events = record['invocation_events']
    repeated = [row for row in events if row['module'] == 'first']
    assert len(repeated) == 2 and repeated[0]['call_id'] != repeated[1]['call_id']
    load = lambda ref: np.load(destination / 'capture' / ref['blob_file'], allow_pickle=False)
    assert np.array_equal(load(repeated[0]['output']), original.numpy())
    assert np.array_equal(load(repeated[1]['output']), original.clamp_min(0).numpy())
    assert record['registered_modules']['unused']['actual_calls'] == 0
    relu = next(row for row in events if row['module'] == 'relu')
    assert relu['output']['blob_file'] == repeated[1]['output']['blob_file']
    nested = next(row for row in events if row['module'] == 'model')['output']
    bf16 = nested['entries'][0]['value']['items'][1]['items'][0]
    assert bf16['encoding'] == 'bfloat16_raw_bits_as_uint16'
    assert np.array_equal(load(bf16), expected['nested'][1][0].view(torch.uint16).numpy())
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    result = dict(status='PASS_ACTUAL_CPU_RAW_RECORDER_FIXTURE',
        engineering_fixture_forward_calls=2, candidate_model_forwards=0,
        baseline_model_forwards=0, optimizer_updates=0, epochs_added=0,
        repeated_calls_separate=True, pre_inplace_output_preserved=True,
        bfloat16_bits_exact=True, output_parity_exact=True, hooks_removed=True,
        called_module_events=len(events), unique_blobs=len(record['unique_physical_blobs']),
        GPU_model_execution_not_tested=True, detection_efficacy_not_tested=True)
    write(destination / 'verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    main(parser.parse_args().destination)
