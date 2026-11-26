"""Validate the completed phase-5 documents and freeze their exact evidence."""
import sys,importlib,re,ast,json,math,collections
sys.stdout.reconfigure(encoding='utf-8');C=importlib.import_module('342_stage5_context')

def main():
    p=C.HERE/'342_开训前审查包.md';src=p.read_text(encoding='utf-8')
    result=C.read(C.HERE/'342_阶段5_result_v2.json');assert C.sha(p)==result['package_sha256']
    headings=re.findall(r'^## ([1-6])\.',src,re.M);assert headings==list('123456')
    docs=[p,C.HERE/'342_阶段5_中文汇总_v2.md'];refs=[];table_checks=0
    for doc in docs:
        body=doc.read_text(encoding='utf-8')
        for path in re.findall(r'\]\(([^)]+)\)',body):
            pp=C.Path(path);assert pp.is_absolute() and pp.is_file(),path;refs.append(str(pp))
        expected_cols=None
        for line in body.splitlines():
            if line.startswith('|'):
                n=len(re.split(r'(?<!\\)\|',line))-2
                if expected_cols is None:expected_cols=n
                assert n==expected_cols,('table columns',doc.name,line)
                table_checks+=1
            else:expected_cols=None
        assert not re.search(r'python[^\n]*(--device[^\n]*cuda|torchrun|--epochs)',body,re.I)
        assert '可行' not in body and '不可行' not in body
    manifest=C.read(C.HERE/'342_活动缓存7566_逐条SHA总清单.json');assert len(manifest['entries'])==7566
    plans=C.read(C.HERE/'342_八run配置清单.json')['runs'];assert len(plans)==8
    for r in plans:
        assert r['total_steps']==1218 and r['total_crop_exposures']==19410 and r['last_batch_size']==3
        assert r['global_state_key']=='model' and r['EMA'] is False
    T=importlib.import_module('train344');lr=C.read(C.HERE/'342_预定学习率曲线1218点.json')
    assert lr['non_kappa']==[T.lr_at(i,1218) for i in range(1218)]
    assert lr['G_kappa']==[T.lr_at_kappa(i,1218) for i in range(1218)]
    ledger=C.read(C.HERE/'342_审查包证据文件SHA256_v2.json')
    for path,h in ledger['files'].items():assert C.sha(path)==h
    inactive=['342_build_review_package.py'] # Intentionally preserved pre-execution syntax failure.
    active=[]
    for path in C.HERE.rglob('*.py'):
        if path.name in inactive:continue
        ast.parse(path.read_text(encoding='utf-8'));active.append(str(path))
    C.write('342_result_阶段5交付核查_v2.json',{'status':'已验证（文档/链接/配置/身份）','six_sections':headings,
        'package_path':str(p),'package_sha256':C.sha(p),'unique_local_link_count':len(set(refs)),
        'inactive_failed_source_preserved':inactive,'active_source_AST_paths':active,
        'syntax_failure_receipt':str(C.HERE/'342_result_审查包生成首次语法实现失败.json'),
        'count_correction_receipt':str(C.HERE/'342_result_阶段5合计更正.json'),'已执行':{'local_file_link_checks':len(refs),'table_row_checks':table_checks,'run_config_checks':8,
            'LR_value_bit_checks':2436,'evidence_SHA_checks':len(ledger['files']),'active_AST_parses':len(active),
            'model_forwards':0,'optimizer_steps':0},
        '未执行':['训练','GPU','SSH','队列'],'未核':result['未核'],'source_sha256':C.sources()})
    C.text('342_阶段5_交接索引.md','\n'.join([
        '# 342 阶段5交接索引','',
        f'主审查包：[342_开训前审查包.md]({p.as_posix()})。阶段5文件已完成，停下等主控审查。真实训练未授权、未执行。','',
        f'- [一页中文汇总]({(C.HERE/"342_阶段5_中文汇总_v2.md").as_posix()})',
        f'- [阶段5 result与调用数]({(C.HERE/"342_阶段5_result_v2.json").as_posix()})',
        f'- [20项合同证据表]({(C.HERE/"342_合同逐条证据表.json").as_posix()})',
        f'- [development/val 7566条SHA]({(C.HERE/"342_活动缓存7566_逐条SHA总清单.json").as_posix()})',
        f'- [八run固定配置]({(C.HERE/"342_八run配置清单.json").as_posix()})',
        f'- [当前真实头CPU计量]({(C.HERE/"342_result_CPU单窗资源实测_v2.json").as_posix()})',
        f'- [文档与证据核查]({(C.HERE/"342_result_阶段5交付核查_v2.json").as_posix()})',
        f'- [最终封存SHA256]({(C.HERE/"342_阶段5_交付封存SHA256.json").as_posix()})','',
        '未核项与正式入口缺口已在审查包第6节列明。训练后的G Δ/八run指标属于未执行的后续工作，不冒称预检已证明科学效果。','']))
    files=[p]+[q for q in C.HERE.rglob('*') if q.is_file()]
    artifacts={str(q):{'sha256':C.sha(q),'bytes':q.stat().st_size} for q in sorted(files)}
    C.write('342_阶段5_交付封存SHA256.json',{'status':'阶段5交付，待主控；真实训练未授权','artifacts':artifacts,
        '已执行':{'artifacts_sealed':len(artifacts),'real_training_steps':0,'optimizer_steps':0,'GPU_calls':0},
        '未执行':['真实训练','GPU','SSH','队列'],'未核':result['未核'],'source_sha256':C.sources()})
    print(json.dumps({'status':'PASS; stop for review','package':str(p),'contracts':20,'run_configs':8,'cache_SHA_entries':7566,
        'sealed_artifacts':len(artifacts),'local_link_checks':len(refs),'LR_checks':2436},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
