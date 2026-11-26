"""The single rejected raw candidate: independent coordinates, six visual zooms."""
import importlib,json
C=importlib.import_module('343_context');G=C.load_geometry(C.HERE/'343_geometry_v2/geom342.py')
import torch,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    source=C.WORK/'340_cache_seed2/001466.pt';old=torch.load(source,map_location='cpu',weights_only=True)
    prior=torch.load(C.WORK/'339_cache_seed2/001466.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(old[k],prior[k]) for k in ('final300_boxes','final300_logits'))
    ow,oh=old['orig_wh'];edges=G.global_queries_to_orig_xyxy(old['final300_boxes'].double(),[ow,oh])[62].numpy()
    cx,cy,w,h=map(float,old['final300_boxes'][62]);ref=np.array([cx*ow-w*ow/2,cy*oh-h*oh/2,cx*ow+w*ow/2,cy*oh+h*oh/2])
    assert np.max(np.abs(edges-ref))<1e-8 and edges[3]<edges[1]
    try:G._boxes(G.raw_query_cxcywh_to_640_xyxy(old['final300_boxes'])[62:63])
    except AssertionError:rejected=True
    else:raise AssertionError('nonpositive candidate did not fail')
    src=C.Path(r'C:\Users\jiwei\Desktop\DEIM\dataset_visdrone\VisDrone2019-DET-train\images')/old['file_name']
    with Image.open(src) as im:pixels=np.asarray(im.convert('RGB'))
    window=G.plane640_to_orig_xyxy(old['window_xyxy_640'][None],[ow,oh])[0].numpy()
    paths=[]
    for number,width in enumerate((1400,400,200,120,80,40)):
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for ax,b,title in zip(axes,(edges,ref),('Shared raw-query conversion','Independent center / extent conversion')):
            ax.imshow(pixels)
            x0,y0,x1,y1=b
            ax.plot([x0,x1],[y0,y0],color='lime',lw=2,label=f'declared top y={y0:.6f}')
            ax.plot([x0,x1],[y1,y1],color='red',lw=2,label=f'declared bottom y={y1:.6f}')
            ax.plot([x0,x0,x1,x1],[y0,y1,y1,y0],color='cyan',lw=1)
            wx,wy,wr,wb=window;ax.plot([wx,wr,wr,wx,wx],[wy,wy,wb,wb,wy],color='yellow',lw=1)
            mx,my=cx*ow,cy*oh
            ax.set_xlim(mx-width/2,mx+width/2);ax.set_ylim(my+width*.375,my-width*.375)
            ax.set_title(title);ax.legend(fontsize=7,loc='upper right')
        fig.suptitle(f'seed2 development1466 / query62 class3 score=.012048937; NEGATIVE HEIGHT; visual zoom only ({width}px)')
        fig.tight_layout();name=f'343_非正框1466_目视_{number}.png'
        with open(C.HERE/name,'xb') as f:fig.savefig(f,format='png',dpi=120)
        plt.close(fig);paths.append(str(C.HERE/name))
    C.write('343_result_3b非正框独立几何与PNG.json',{'status':'已验证该条必须触发当前非正框合同；数据集仍实现失败',
        'seed':2,'image_id':1466,'query':62,'candidate_class':3,'score':float(old['fullp_scores'][259]),
        'raw_normalized_cxcywh':old['final300_boxes'][62].tolist(),'original_xyxy':edges.tolist(),
        'independent_original_xyxy':ref.tolist(),'independent_max_abs_diff':float(np.max(np.abs(edges-ref))),
        'reverse_nonpositive_xyxy_rejected':rejected,'raw_equal_to_339':True,'png_paths':paths,
        'note':'6图是同一坏条目的6个显示倍率，不是6个坏条目；显示倍率不改变冻结选窗、坐标或标签。黄色为原冻结W。',
        '已执行':{'old_cache_loads':2,'source_image_decodes':1,'independent_geometry_comparisons':1,'negative_input_tests':1,'PNG生成':6,
            'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['修正/过滤坏框','该条新M0生成','真实训练','阶段5'],'未核':['主控独立复核'],'source_sha256':C.sources()})

if __name__=='__main__':main()
