"""Plot verified saved logs only; this process never imports Torch or accesses a model."""
from pathlib import Path
import argparse,hashlib,json,math,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ORDER=('pretrained_BASE','original_stem','native_stem_scratch','detail_lowres','highfreq_rgb')
LABELS={'pretrained_BASE':'Pretrained BASE (cached)','original_stem':'EnhancedStem (full)',
        'native_stem_scratch':'OriginalStem (random)','detail_lowres':'Detail at half resolution','highfreq_rgb':'RGB high-frequency input'}
COLORS={'pretrained_BASE':'#545c66','original_stem':'#18885d','native_stem_scratch':'#2563bb',
        'detail_lowres':'#d38b20','highfreq_rgb':'#bd4262'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(data_path,output):
    data=json.loads(Path(data_path).read_bytes());rows=data['rows']
    assert data['status']=='VERIFIED_SAVED_LOG_CURVES_FIXED12_PENDING'
    assert len({(r['seed'],r['arm'],r['epoch']) for r in rows})==len(rows)
    assert all(r['seed'] in (1,2) and r['arm'] in ORDER and 1<=r['epoch']<=12 and
               math.isfinite(r['AP']) and math.isfinite(r['APs']) for r in rows)
    assert data['new_model_forwards']==data['new_optimizer_updates']==0
    out=Path(output);out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','axes.titleweight':'semibold'})
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.4),sharex=True)
    limits={metric:math.ceil(max(r[metric] for r in rows)*1.08) for metric in ('AP','APs')}
    handles=[];labels=[];plotted=[]
    for ri,seed in enumerate((1,2)):
        for ci,metric in enumerate(('AP','APs')):
            ax=axes[ri,ci];ax.set_title(f'Seed {seed} | '+('AP' if metric=='AP' else 'AP small'))
            for arm in ORDER:
                points=sorted([r for r in rows if r['seed']==seed and r['arm']==arm],key=lambda r:r['epoch'])
                if not points:continue
                assert [r['epoch'] for r in points]==list(range(1,len(points)+1))
                line,=ax.plot([r['epoch'] for r in points],[r[metric] for r in points],
                    color=COLORS[arm],linewidth=2.0 if arm=='original_stem' else 1.65,marker='o',
                    markersize=4 if arm=='original_stem' else 3,linestyle='--' if arm=='pretrained_BASE' else '-',
                    alpha=.98 if arm in ('original_stem','native_stem_scratch') else .85)
                if ri==ci==0:handles.append(line);labels.append(LABELS[arm])
                if arm=='original_stem':
                    last=points[-1];ax.annotate(f"{last[metric]:.3f}",(last['epoch'],last[metric]),
                        xytext=(7,5),textcoords='offset points',color=COLORS[arm],fontsize=9)
                plotted.append(dict(seed=seed,metric=metric,arm=arm,epochs=[p['epoch'] for p in points],
                                    values=[p[metric] for p in points]))
            ax.axvline(12,color='#999999',linewidth=.8,linestyle=':')
            ax.set_xlim(.6,12.7);ax.set_ylim(0,limits[metric]);ax.set_xticks(range(1,13))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6));ax.set_ylabel(metric+' (%)')
            ax.grid(axis='y',color='#dde2e7',alpha=.8,linewidth=.7)
            if ri==1:ax.set_xlabel('Completed epoch')
    fig.suptitle('Stem ablations: all available epochs',x=.055,y=.98,ha='left',fontsize=17,fontweight='bold')
    fig.text(.055,.943,'Same native S recipe and seed; outside Stem shares the same initial weights. BASE uses existing logs.',fontsize=10,color='#425064')
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.916),ncol=3,frameon=False,fontsize=9.5)
    fig.subplots_adjust(left=.075,right=.975,bottom=.13,top=.805,wspace=.19,hspace=.27)
    fig.text(.055,.062,'Only completed epochs are shown. Full/reference training endpoints and independent epoch12 EMA verification remain pending.',
             fontsize=9,color='#425064')
    fig.text(.055,.038,'The two remaining controls have no training results yet. No curves are extrapolated; AP and AP small use their native COCO definitions.',
             fontsize=8.7,color='#425064')
    for suffix in ('png','svg','pdf'):
        path=out/('stem_curves_v1.'+suffix)
        fig.savefig(path,dpi=150,facecolor='white',metadata={'Creator':'Verified saved-log plotting'} if suffix=='pdf' else None)
    plt.close(fig)
    assert 'torch' not in sys.modules
    report=dict(status='RENDERED_VERIFIED_SAVED_LOG_CURVES',data_sha256=sha(data_path),plot_code_sha256=sha(__file__),
        source_rows=len(rows),plotted_series=plotted,epochs_added=0,model_forwards=0,
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()},
        interpretation=data['scope'])
    with (out/'plot_report.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=report['status'],source_rows=len(rows),plot_code_sha256=sha(__file__)),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.data,a.output)

