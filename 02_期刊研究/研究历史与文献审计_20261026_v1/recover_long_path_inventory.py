import ast,json,os,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent.parent
inv=json.loads((OUT/'filesystem_inventory.json').read_text(encoding='utf-8'))
records=[]
for err in inv['errors']:
    if '[WinError 3]' not in err:continue
    try:
        quoted=err[err.index(': ')+2:]
        p=Path(ast.literal_eval(quoted))
        assert p.is_absolute() and p.is_relative_to(ROOT)
        extended='\\\\?\\'+str(p)
        entries=list(os.scandir(extended))
        records.append({'path':str(p),'readable_with_extended_path':True,
                        'entry_names':[x.name for x in entries],
                        'note':'Metadata only; raw conversation bodies not fully reviewed.'})
    except Exception as e:
        records.append({'original_error':err,'recovery_error':str(e),'readable_with_extended_path':False})
summary={'initial_walk_errors':len(inv['errors']),'long_path_recoveries':len(records),
         'successful':sum(r['readable_with_extended_path'] for r in records),
         'remaining_permission_errors':sum('[WinError 5]' in e for e in inv['errors']),
         'records':records}
(OUT/'long_path_scope_recovery.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False))
