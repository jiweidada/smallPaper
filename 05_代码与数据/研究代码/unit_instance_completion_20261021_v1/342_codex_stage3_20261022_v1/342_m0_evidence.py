"""Read-only independent M0 class audit and six review PNGs. No relabeling."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,importlib
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;WORK=HERE.parent
PREV=WORK/'342_codex_pretrain_20261022_v1'
sys.path.insert(0,str(PREV))
import torch,numpy as np
from PIL import Image,ImageDraw
import geom342 as G
torch.set_num_threads(2)
VAL=Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-val')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    prior=json.loads((HERE/'342_result_阶段3输入预检.json').read_text(encoding='utf-8'))
    rows=[];local=0;pngs=[];maxerr=0.;reverse_rejected=0
    for row in prior['rows']:
        e=torch.load(row['cache_path'],map_location='cpu',weights_only=True)
        # Different implementation from preflight: masked vector lookup in
        # NumPy instead of per-candidate Python/Torch scalar comparisons.
        owners=e['m0_cand_owner'].numpy();cand=e['fullp_labels'].numpy();gt=e['gt_labels'].numpy()
        matched=np.flatnonzero(owners>=0)
        wrong=matched[cand[matched]!=gt[owners[matched]]]
        assert wrong.tolist()==[r['candidate_index'] for r in row['wrong_class_owners']]
        details=[]
        for j in wrong:
            g=int(owners[j]);inside=bool(e['G_W'][g]);local+=int(inside)
            details.append({'candidate_index':int(j),'candidate_label':int(cand[j]),'owner_gt_index':g,
                'owner_gt_label':int(gt[g]),'owner_in_G_W':inside,'query_argmax_label':int(e['fullp_logits_row'][j].argmax())})
        rows.append({'image_id':e['image_id'],'wrong_class_count':len(wrong),'wrong_owners':details})
        if len(wrong) and len(pngs)<6:
            with Image.open(VAL/'images'/e['file_name']) as f:pil=f.convert('RGB')
            ow,oh=pil.size
            dw=1024;dh=round(1024*oh/ow)
            pic=pil.resize((dw,dh),Image.BILINEAR)
            canvas=Image.new('RGB',(dw,dh+112),'white');canvas.paste(pic,(0,96))
            d=ImageDraw.Draw(canvas)
            j=int(wrong[0]);g=int(owners[j]);W=torch.tensor([0.,0.,640.,640.])
            b=torch.stack((e['fullp_boxes'][j],e['gt_boxes_640'][g],e['window_xyxy_640']))
            edges=G.win_norm_to_window_pixels(G.win_norm_xyxy(b,W),W,(dw,dh)).numpy()
            # Independent scalar 640 -> display check does not call geom342.
            oracle=[]
            for bb in b.numpy():oracle.append([float(bb[0])*dw/640,float(bb[1])*dh/640,float(bb[2])*dw/640,float(bb[3])*dh/640])
            err=float(np.abs(edges-np.array(oracle)).max());maxerr=max(maxerr,err)
            assert err<1e-3
            try:G.win_norm_xyxy(G.xyxy_to_cxcywh(b[:2]),W)
            except AssertionError:reverse_rejected+=1
            else:
                bad=G.win_norm_to_window_pixels(G.win_norm_xyxy(G.xyxy_to_cxcywh(b[:2]),W),W,(dw,dh)).numpy()
                assert np.max(np.abs(bad-edges[:2]))>1
            for edge,color in zip(edges,('#00ffff','#ff3030','#00ff00')):
                rect=edge+[0,96,0,96];d.rectangle(tuple(rect),outline=color,width=3)
            d.text((8,8),f'image_id={e["image_id"]}; candidate #{j} query #{int(e["fullp_query_index"][j])}',fill='black')
            d.text((8,27),f'candidate instance class={int(cand[j])}; query argmax={int(e["fullp_logits_row"][j].argmax())}',fill='black')
            d.text((8,46),f'owner GT class={int(gt[g])}; annotation_id={e["gt_annotation_ids"][g]}; owner in G_W={bool(e["G_W"][g])}',fill='black')
            d.text((8,65),'Cyan=candidate, red=assigned GT, green=window. CLASS MISMATCH; old cache preserved.',fill='black')
            name=f'342_M0跨类证据_{int(e["image_id"]):06d}.png'
            with open(HERE/name,'xb') as f:canvas.save(f,format='PNG')
            pngs.append(str(HERE/name))
    report={'status':'实现失败（继承的M0候选类别合法边错误）','independent_wrong_class_total':sum(r['wrong_class_count'] for r in rows),
        'wrong_class_owner_in_G_W':local,'rows':rows,'png_paths':pngs,
        'display_geometry_independent_max_abs_diff':maxerr,'wrong_format_assert_rejected_PNG_cases':reverse_rejected,
        'source_lines':[{'path':str(WORK/'uic_m0.py'),'line':53,'code':'cand_cls=cl.argmax(-1)'},
                        {'path':str(WORK/'uic_m0.py'),'line':56,'code':'valid uses cand_cls==gt_label, not fullP candidate label'}],
        'additional_static_concern':{'path':str(WORK/'uic_m0.py'),'lines':[78,80,84,90,91],
            'observation':'三级lex用了epsilon*(cand_index*M+gt_index)，并按1e-6比较cost变化；尚未证明符合296要求的精确owner向量lex。此项未做新的数值结论。'},
        'source_sha256':{str(p):sha(p) for p in (Path(__file__),WORK/'uic_m0.py',WORK/'stage1_cache_v3.py',PREV/'geom342.py',HERE/'342_stage3_preflight.py')},
        '已执行':{'cache_loads':32,'independent_class_vector_checks':32,'PNG生成':6,'global_forwards':0,'optimizer_steps':0},
        '未执行':['M0修复','标签改写','完整目标缓存构建','真实训练','GPU','SSH','heldout3236读取','阶段4至5'],
        '未核':['精确三级lex合同','修复M0后对全部目标的影响'],'待主控':['是否修复M0并按字段披露与旧341缓存的差异']}
    with open(HERE/'342_result_M0独立核查.json','x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('independent_wrong_class_total','wrong_class_owner_in_G_W','display_geometry_independent_max_abs_diff')},ensure_ascii=False))

if __name__=='__main__':main()
