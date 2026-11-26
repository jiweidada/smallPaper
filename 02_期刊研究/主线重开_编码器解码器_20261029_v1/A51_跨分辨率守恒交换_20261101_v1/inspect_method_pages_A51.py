from pathlib import Path
import json,re,subprocess
ROOT=Path(__file__).resolve().parent
LIB=ROOT.parent.parent/'文献_相关原始论文_20261004'
specs=[('A51_mHC_arXiv2512.24880v1','4.1. Manifold-Constrained'),
 ('A51_FreqFusion_arXiv2408.12879v1','4 F REQUENCY - AWARE F EATURE F USION'),
 ('A51_RevBiFPN_arXiv2206.14098v1','The Reversible Residual Silo'),
 ('A51_NeuralSheafDiffusion_arXiv2202.04579v1','Xt+1 = Xt')]
for stem,marker in specs:
    pages=(ROOT/'selected_text'/(stem+'.txt')).read_text(encoding='utf-8').split('\f')
    found=[i+1 for i,t in enumerate(pages) if marker in re.sub(r'\s+',' ',t)]
    assert found,(stem,marker)
    page=found[0]
    prefix=ROOT/(stem+'_method_p'+str(page))
    subprocess.run([r'C:\texlive\2025\bin\windows\pdftoppm.exe','-f',str(page),'-l',str(page),'-r','90','-singlefile','-png',str(LIB/(stem+'.pdf')),str(prefix)],check=True)
    print(json.dumps({'stem':stem,'page':page,'image':str(prefix)+'.png'}))
