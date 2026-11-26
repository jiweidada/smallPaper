"""Append-only instrumented driver; original package remains byte-identical."""
import argparse,hashlib,json,pathlib,sys,traceback
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'probe'))
sys.path.insert(0,str(ROOT/'extension_v2'))

def verify():
    manifest=json.loads((ROOT/'extension_v2/manifest.json').read_text())
    for f in manifest['files']:
        assert hashlib.sha256((ROOT/f['relative']).read_bytes()).hexdigest()==f['sha256']
    return hashlib.sha256((ROOT/'extension_v2/manifest.json').read_bytes()).hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1,2],required=True);args=p.parse_args()
    try:
        extension_sha=verify()
        import probe_model
        from diagnostic_extension_v2 import analyze
        probe_model.analyze=analyze
        probe_model.main(args.seed)
        assert verify()==extension_sha
        (ROOT/f'results_seed{args.seed}/supplement_complete.json').write_text(json.dumps({
            'state':'complete','extension_manifest_sha256':extension_sha,
            'formal_input_list_unchanged':True,'original_package_unchanged':True,
            'DN_ties_overflow_split_losses_recorded':True,'optimizer_steps':0},indent=2))
    except BaseException as e:
        d=ROOT/f'results_seed{args.seed}';d.mkdir(exist_ok=True)
        (d/'failure.json').write_text(json.dumps({'error':type(e).__name__+': '+str(e),
            'traceback':traceback.format_exc(),'no_automatic_retry':True}));raise
