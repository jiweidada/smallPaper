from pathlib import Path
from copy import deepcopy
import hashlib
import json
import shutil
import sys
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

sys.stdout.reconfigure(encoding='utf-8')
source = Path(r'C:\Users\jiwei\Desktop\大论文\大论文框架\大论文框架.docx')
work = Path(__file__).resolve().parent
expected_sha = 'ed691f0f50329f45327ad68383285735a3fa43beaa38c2466d51813e83d11a2b'
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
if source_sha != expected_sha:
    raise RuntimeError('The source document changed; inspect the latest version before editing.')
backup = work / '大论文框架_修改前备份.docx'
if not backup.exists():
    shutil.copy2(source, backup)
elif hashlib.sha256(backup.read_bytes()).hexdigest() != source_sha:
    raise RuntimeError('Existing backup differs from the current source.')

doc = Document(source)
changes = []

def replace(old, new):
    matches = [p for p in doc.paragraphs if p.text == old]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one paragraph matching {old!r}; found {len(matches)}.')
    p = matches[0]
    p.text = new
    changes.append({'before': old, 'after': new})
    return p

replace('围绕无人机航拍图像中小目标细节不足、尺度变化和低照度干扰等问题，研究小目标特征表达与场景适应方法，并通过检测系统验证其应用效果。',
        '围绕无人机航拍图像中小目标细节不足、尺度变化大等问题，研究细节保留与多尺度上下文建模方法，通过实验分析评价检测性能，并完成检测系统的设计与验证。')
replace('研究内容二：拟研究低照度条件下的小目标检测适应性，分析已有模型的漏检与定位误差，比较常规增强策略，并在此基础上确定改进方法。',
        '研究内容二：围绕 MPSMA-DEIM 开展实验验证与性能分析，在统一对比条件下分析各模块的作用、小目标检测表现及模型复杂度与推理效率，拟补充不同数据或场景下的适用性验证。')
replace('第2章 相关理论与实验基础', '第2章 相关理论与技术基础')
replace('2.4 低照度成像与目标检测', '2.4 目标检测训练与损失函数')
replace('2.5 数据集 评价指标与实验协议', '2.5 检测精度与计算效率评价方法')
replace('3.4 整体模型与训练设置', '3.4 整体网络与训练目标')
replace('3.5 消融实验与方法对比', '3.5 本章小结')
for old in ['3.6 模型复杂度 推理效率与结果分析', '3.7 本章小结']:
    matches = [p for p in doc.paragraphs if p.text == old]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one paragraph matching {old!r}.')
    p = matches[0]
    p._element.getparent().remove(p._element)
    changes.append({'before': old, 'after': None})

replace('第4章 低照度条件下无人机小目标检测的适应性研究',
        '第4章 MPSMA-DEIM的实验验证与性能分析')
replace('4.1 低照度条件下的检测失效分析', '4.1 数据集与实验设置')
replace('4.2 低照度数据与评估协议', '4.2 整体性能对比')
replace('4.3 常规适应策略及其局限', '4.3 模块消融与作用分析')
replace('4.4 面向检测任务的适应方法', '4.4 小目标检测表现与失效分析')
replace('4.5 对比实验 消融实验与适用范围分析', '4.5 模型复杂度与推理效率')
p = replace('4.6 本章小结', '4.6 不同数据或场景下的适用性验证')
new_element = deepcopy(p._p)
p._p.addnext(new_element)
new_p = Paragraph(new_element, p._parent)
new_p.text = '4.7 本章小结'
changes.append({'before': None, 'after': new_p.text})

text = '\n'.join(p.text for p in doc.paragraphs)
assert '低照度' not in text
assert '拟补充不同数据或场景下的适用性验证' in text
assert '三 当前进展' not in text
assert len([p for p in doc.paragraphs if p.style.name == 'Heading 2']) == 6
assert all(text.count(f'{i}.1 ') == 1 for i in range(1, 7))
assert len(doc.tables) == 0
for tag in ('ins', 'del'):
    assert not doc.element.findall('.//' + qn('w:' + tag)), 'Unexpected tracked changes.'

output = work / '大论文框架.docx'
doc.save(output)
(work / '修改后正文.txt').write_text(text + '\n', encoding='utf-8')
manifest = {'client_date': '2026-11-02', 'source': str(source), 'source_sha256': source_sha,
            'backup': str(backup), 'output': str(output),
            'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'changes': changes, 'format': 'Preserved source styles, section dimensions and explicit page break.',
            'pending_validation': 'Different-data/scenario validation remains proposed in the research-content paragraph.',
            'rendering': 'Use Word PDF export and Poppler; bundled renderer previously diagnosed as lacking LibreOffice.'}
(work / '修改记录.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'output': str(output), 'source_sha256': source_sha,
                  'output_sha256': manifest['output_sha256'], 'changed_items': len(changes)}, ensure_ascii=False))
