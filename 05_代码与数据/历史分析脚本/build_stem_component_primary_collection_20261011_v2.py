"""New component-specific primary collection contract; old collection remains frozen."""
from pathlib import Path
import ast,base64,hashlib,json
root=Path('英文小论文/code/stem_component_primary_collection_20261011_v2');root.mkdir(exist_ok=False)
source=Path('英文小论文/code/stem_primary_collection_20261011_v1/collect.py').read_text()
def change(a,b):
    global source
    assert a in source,a[:180]
    source=source.replace(a,b)
change("    rootname={'original_stem':'original_stem12_20261011_v1','native_stem_scratch':'native_stem_scratch12_20261011_v3'}[arm]", "    assert arm in ('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample')\n    rootname='stem_component12_20261011_v1'")
a=source.index("    protocol=base/'native_selection_diagnostic12_20261011_v1");b=source.index('    sys.path.insert(0,str(code))',a)
source=source[:a]+'''    collector_root=Path(__file__).resolve().parent
    protocol=read(collector_root/'protocol.json')
    assert sha(__file__)==protocol['collector_sha256']
    code=base/rootname/'code'
    assert {p.name:sha(p) for p in code.iterdir() if p.is_file()}==protocol['component_files_sha256']
'''+source[b:]
change("base/'stem_primary_collection_20261011_v1'", "base/'stem_component_primary_collection_20261011_v2'")
change("schema='exact12_primary_artifact_collection_v1'", "schema='exact12_component_primary_artifact_collection_v2'")
change("choices=('original_stem','native_stem_scratch')", "choices=('detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample')")
change("        collector_sha256=sha(__file__),new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)", "        collector_sha256=sha(__file__),collector_protocol_sha256=sha(collector_root/'protocol.json'),\n        new_model_forwards=0,new_evaluator_calls=0,new_training_epochs=0)")
ast.parse(source)
(root/'collect.py').write_text(source,encoding='utf-8',newline='\n')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=dict(schema='component_primary_collection_v2',collector_sha256=sha(root/'collect.py'),
    component_files_sha256=json.loads(Path('outputs/stem_component12_20261011_v1/local_code_manifest.json').read_bytes()),
    allowed_arms=['detail_lowres','highfreq_rgb','attention_spatial_mean','early_final_downsample'],
    completed_epochs_required=12,trainer_and_controller_handles_required_stopped=True,
    new_model_forwards=0,new_evaluator_calls=0,new_optimizer_updates=0,
    independent_saved_EMA_replay_included=False,
    scope='Collect actual fixed12 primary native validation weights, per-GT flags and548 stage arrays from component controls only. Neither a saved-weight replay nor a formal terminal reader pass. Old full/reference collector v1 is unchanged.')
(root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
files={p.name:dict(sha256=sha(p),base64=base64.b64encode(p.read_bytes()).decode()) for p in root.iterdir() if p.is_file()}
print(json.dumps(files))
