"""Preserve the seed2 verifier and derive a generic verifier using saved local stages."""
from pathlib import Path

p=Path(__file__).parent
s=(p/'verify_downloaded_raw_seed2_v1.py').read_text(encoding='utf8')
s=s.replace('import hashlib\n','import argparse\nimport hashlib\n',1)
s=s.replace("DEST=ROOT/'outputs/candidate_all_module_capture_20261010_v1/seed2_download_v1'\n",'')
s=s.replace('def main():\n',"def main(seed):\n    DEST=ROOT/f'outputs/candidate_all_module_capture_20261010_v1/seed{seed}_download_v1'\n    target=ROOT/f'outputs/scale_protection12_20261010_v1/terminal_collection/seed{seed}_target_protect_v1'\n    target_report=json.loads((target/'verification_report_v1.json').read_text())\n    assert target_report['status']=='PASS_FIXED12_DOWNLOADED_EVIDENCE_AND_SAVED_WEIGHT_RELOAD'\n    assert target_report['identity']['seed']==seed and target_report['identity']['arm']=='target_protect'\n    stagepath=target/'extracted/run/stages_epoch12'\n    stage_manifest=json.loads((stagepath/'manifest.json').read_text())\n")
s=s.replace("DEST/'module_candidate_panel_seed2_raw_v1.zip'","DEST/f'module_candidate_panel_seed{seed}_raw_v1.zip'")
s=s.replace("DEST/'module_candidate_panel_seed2_raw_v1_metadata.json'","DEST/f'module_candidate_panel_seed{seed}_raw_v1_metadata.json'")
s=s.replace("    assert meta['sha256']=='4d25184e2615ad01ebb36fbb6da1ef6241fafc28c953787666da011cb2aeddfa'\n", "    assert len(meta['sha256'])==64 and meta['actual_capture_exit_code']==0\n")
s=s.replace("with zipfile.ZipFile(archive) as z,zipfile.ZipFile(DEST/'module_candidate_FDR_subset_seed2_v1.zip') as subset:","with zipfile.ZipFile(archive) as z:")
s=s.replace("==meta['manifest_sha256']=='38d9a79310ee73e75d4406b76f85a9017d2afb12e27507b0c0d724eecec3720d'","==meta['manifest_sha256']")
s=s.replace("        manifest=json.loads(raw)\n", "        manifest=json.loads(raw)\n        for name,key in [('raw_module_recorder.py','recorder_source_sha256'),('capture_candidate_panel.py','capture_source_sha256')]:\n            assert hashlib.sha256(z.read('source/'+name)).hexdigest()==manifest[key]\n")
s=s.replace("z.read('control/module_candidate_completion_seed2_v1.json')","z.read(f'control/module_candidate_completion_seed{seed}_v1.json')")
s=s.replace("        assert verification==completion['verification']\n", "        assert verification==completion['verification']\n        assert verification['seed']==seed and verification['identity']==target_report['identity']\n        expected={1:'e195f5a3d126bee9e0f7a373f5eb5fdebb78ada9fc891780e251be9dafc2ae94',2:'8113eb35c4ca0519e9757a101327cb95e709ebd3609559f55c240e1a3a472269'}\n        assert verification['checkpoint_sha256']==expected[seed]\n        assert verification['protocol_sha256']=='f251e2a4c8ebd4c4737d46a76aa71aede595d346d2e19b011bf60437de9a3fc9'\n")
s=s.replace("                with np.load(io.BytesIO(subset.read(f'stages/image_{iid:06d}.npz')),allow_pickle=False) as stage:","                file=stagepath/f'image_{iid:06d}.npz'\n                assert sha(file)==stage_manifest['files'][file.name]\n                with np.load(file,allow_pickle=False) as stage:")
s=s.replace("OUTPUTS',seed=2,","OUTPUTS',seed=seed,")
s=s.replace("if __name__=='__main__':main()", "if __name__=='__main__':\n    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(1,2),required=True)\n    main(parser.parse_args().seed)")
assert 'subset' not in s and 'seed=2' not in s
compile(s,'verify_downloaded_raw_v2.py','exec')
with (p/'verify_downloaded_raw_v2.py').open('x',encoding='utf8',newline='\n') as f:f.write(s)
print('Saved generic raw verifier v2; original v1 unchanged.')
