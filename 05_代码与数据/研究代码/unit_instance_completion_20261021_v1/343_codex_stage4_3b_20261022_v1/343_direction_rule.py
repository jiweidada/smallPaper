"""317 section 7: twelve strict directions, with no averaging."""
import importlib,math,copy
C=importlib.import_module('343_context')

def judge(metrics,g_effective=None):
    rows=[];overall=[]
    for seed in ('1','2'):
        one=metrics[seed]
        for arm in ('baseline','N','S','G','R'):
            for name in ('AP','APs','B'):
                assert math.isfinite(one[arm][name])
        for control in ('N','S','G'):
            for name in ('APs','B'):
                delta=one['R'][name]-one[control][name]
                rows.append({'seed':int(seed),'control':control,'metric':name,'delta':delta,'strict_positive':delta>0})
        for control in ('baseline','N','S','G'):
            delta=one['R']['AP']-one[control]['AP']
            overall.append({'seed':int(seed),'control':control,'delta_AP':delta,'negative':delta<0})
    keep=all(r['strict_positive'] for r in rows)
    footnotes=[]
    if g_effective is None:footnotes.append('G有效校准状态未核。')
    else:
        for seed in ('1','2'):
            if g_effective[seed] is False:footnotes.append(f'seed{seed} G未有效校准；不能声称已充分排除全局null校准这一解释。')
            elif g_effective[seed] is None:footnotes.append(f'seed{seed} G有效校准状态未核。')
    return {'directions12':rows,'all12_strict_positive':keep,
            'direction_statement':'有限学习方向保留' if keep else '12项方向规则不成立',
            'overall_AP_differences':overall,'footnotes':footnotes,
            'interpretation':'同向筛查；不构成显著性、等效结论或训练授权。'}

def main():
    base={str(seed):{arm:{'AP':1.,'APs':1.,'B':1} for arm in ('baseline','N','S','G','R')} for seed in (1,2)}
    for seed in ('1','2'):base[seed]['R']={'AP':2.,'APs':2.,'B':2}
    cases=[]
    for name,change,expected,g in [
        ('all_positive',None,True,{'1':True,'2':True}),
        ('one_zero',('1','N','APs',2.),False,{'1':True,'2':True}),
        ('one_negative',('2','G','B',3),False,{'1':True,'2':True}),
        ('cross_seed_cancellation',('2','R','APs',0.),False,{'1':True,'2':True}),
        ('G_ineffective_footnote',None,True,{'1':False,'2':False})]:
        m=copy.deepcopy(base)
        if change:
            seed,arm,key,value=change;m[seed][arm][key]=value
        result=judge(m,g);assert len(result['directions12'])==12
        assert result['all12_strict_positive']==expected
        if name=='G_ineffective_footnote':assert len(result['footnotes'])==2
        if name=='cross_seed_cancellation':
            for control in ('N','S','G'):
                assert sum(r['delta'] for r in result['directions12'] if r['control']==control and r['metric']=='APs')==0
        cases.append({'case':name,'constructed_input':m,'expected':expected,'result':result})
    C.write('343_result_阶段4方向规则.json',{'status':'已验证','constructed_cases':cases,
        '已执行':{'judge_calls':5,'constructed_direction_comparisons':60,'global_forwards':0,'head_forwards':0,'optimizer_steps':0},
        '未执行':['八run真实指标判定','真实训练','阶段5'],'未核':['主控独立核对'],
        'source_sha256':C.sources()})

if __name__=='__main__':main()
