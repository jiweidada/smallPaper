"""Build complete uncollapsed native diagrams, explicitly expanding fused SDPA."""
from pathlib import Path
import copy
import hashlib
import json

BASE=Path(__file__).resolve().parents[3]
OUTPUT=BASE/'outputs/model_structure_html_20261010_v1'
CAPTURE=OUTPUT/'collection_complete/captured_complete_graph_v3'


def expand_attention(graph):
    graph=copy.deepcopy(graph);extra=[];removed_edges=set();added_edges=[];expansions=[]
    for n in graph['nodes']:
        if '_scaled_dot_product_' not in n['operator'] or 'attention' not in n['operator']:continue
        raw=copy.deepcopy(n);args=n['scalar_arguments']['args'];q,k,v=args[:3]
        shape=q['shape'][:-1]+[k['shape'][-2]]
        labels=['Q × Kᵀ','Scale · 1/√'+str(q['shape'][-1])]
        mask=len(args)>3 and isinstance(args[3],dict) and args[3].get('tensor')
        if mask:labels.append('Attention mask · DN/normal separation')
        labels+=['Softmax · along keys','Dropout · p=0']
        chain=[]
        for j,label in enumerate(labels):
            identity=n['id']+'_math'+str(j);chain.append(identity)
            extra.append(dict(id=identity,kind='math',operator=label,module=n['module'],outputs=[dict(slot='mathematical value',shape=shape,dtype=q['dtype'])],scalar_arguments=dict(native_fused_operator=raw['operator'],expansion='Mathematical expansion, not separately dispatched kernels'),expanded_from=raw['id']))
        for a,b in zip(chain,chain[1:]):added_edges.append(dict(source=a,target=b,shape=shape,source_slot='mathematical value',target_argument='mathematical input',math=True))
        added_edges.append(dict(source=chain[-1],target=n['id'],shape=shape,source_slot='attention weights',target_argument='A',math=True))
        for i,e in enumerate(graph['edges']):
            if e['target']!=n['id']:continue
            removed_edges.add(i);new=dict(e)
            if e['target_argument'] in ('value.0.0','value.0.1'):new['target']=chain[0]
            elif e['target_argument']=='value.0.2':new['target']=n['id']
            elif mask and e['target_argument']=='value.0.3':new['target']=chain[2]
            else:new['target']=chain[0]
            added_edges.append(new)
        n.update(kind='math',operator='Attention output · A × V',native_operator=raw['operator'],scalar_arguments=raw['scalar_arguments'],expanded_from=raw['id'])
        expansions.append(dict(original_node=raw['id'],native_operator=raw['operator'],module=raw['module'],expanded_nodes=chain+[raw['id']]))
    graph['nodes']+=extra;graph['edges']=[e for i,e in enumerate(graph['edges']) if i not in removed_edges]+added_edges
    graph['attention_math_expansions']=expansions
    return graph


def main():
    original={mode:json.loads((CAPTURE/f'graph_{mode}.json').read_text(encoding='utf8')) for mode in ('train','eval')}
    graphs={mode:expand_attention(g) for mode,g in original.items()}
    report={}
    for mode,g in graphs.items():
        ids={n['id'] for n in g['nodes']};assert len(ids)==len(g['nodes'])
        assert all(e['source'] in ids and e['target'] in ids for e in g['edges'])
        native_ids={n['id'] for n in original[mode]['nodes']};assert native_ids.issubset(ids)
        assert original[mode]['every_called_leaf_invocation_individually_represented']
        covered={n.get('call_id') for n in original[mode]['nodes']}
        leaves=[c for c in original[mode]['module_calls'] if c['leaf']]
        assert all(c['call_id'] in covered for c in leaves)
        report[mode]=dict(native_nodes=len(native_ids),native_operators=sum(n['kind']=='operator' for n in original[mode]['nodes']),native_identity_nodes=sum(n['kind']=='module_identity' for n in original[mode]['nodes']),display_nodes=len(g['nodes']),display_edges=len(g['edges']),registered_modules=len(g['module_inventory']),called_modules=len(g['call_counts']),leaf_invocations=len(leaves),all_leaf_invocations_represented=True,all_native_nodes_retained=True,attention_expansions=len(g['attention_math_expansions']))
    sources={p.relative_to(CAPTURE/'native_source').as_posix():p.read_text(encoding='utf8') for p in (CAPTURE/'native_source').rglob('*.py')}
    data=dict(graphs=graphs,sources=sources,verification=report,checkpoint_sha256=json.loads((CAPTURE/'completion.json').read_text())['checkpoint_sha256'])
    (OUTPUT/'diagram_data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf8')
    (OUTPUT/'diagram_coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__':main()
