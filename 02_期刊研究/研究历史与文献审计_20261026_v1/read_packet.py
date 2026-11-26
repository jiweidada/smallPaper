"""Bounded, line-numbered source reading with a hash ledger."""
import hashlib,json,re,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
folder=ROOT/sys.argv[1]
ledger_path=OUT/'document_read_ledger.json'
ledger=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {}
for name in sys.argv[2:]:
    p=folder/name
    if not p.is_file():
        print('MISSING',p);continue
    b=p.read_bytes();s=b.decode('utf-8-sig',errors='replace');lines=s.splitlines()
    selected=set(range(min(16,len(lines))))
    matches=[i for i,x in enumerate(lines) if re.search('结论|判定|停止|阴性|混合|不支持|未通过|未执行|撤回|主要结果|真实结果|结果：|status|decision|failed|mixed|not_pass',x,re.I)]
    for i in (matches[:8]+matches[-6:]):
        selected.update(range(max(0,i-1),min(len(lines),i+2)))
    selected=sorted(selected)
    ledger[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'total_lines':len(lines),
                    'read_lines':sorted(set(selected+ledger.get(str(p),{}).get('read_lines',[])))}
    print('\nFILE',str(p),'LINES',len(lines))
    for i in selected:
        print(f'{i+1}: {lines[i][:650]}')
ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
