(()=>{
'use strict';
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
const raw=$('network-data').textContent;if(raw.trim().startsWith('___'))return;
const DATA=JSON.parse(raw),svg=$('network'),world=$('world'),vp=$('viewport');
let mode='train',G,view,nodes,moduleMap,groups,owned,subtrees,representatives,nodeElements,treeRows,treeItems;
let expanded=new Set(),treeExpanded=new Set(['model']),selected=null,selectedModule=null,scale=1,tx=0,ty=0,matches=[],matchIndex=0;
const cache=new Map();
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const short=(s,n=44)=>s.length>n?'…'+s.slice(-(n-1)):s;
const shape=s=>s.length?s.join(' × '):'scalar';
const parent=p=>p.includes('.')?p.slice(0,p.lastIndexOf('.')):p==='model'?null:'model';
const ancestors=p=>{const a=[];while(p){a.unshift(p);p=parent(p);}return a;};
const branch=n=>n.module.startsWith('backbone')?'backbone':n.module.startsWith('encoder')?'encoder':n.module.startsWith('decoder')?'decoder':n.kind==='input'?'gt':'muted';
const element=(tag,attrs={},text)=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;};
function name(n){if(n.kind==='group')return n.label;if(['math','module_identity','input'].includes(n.kind))return n.operator;const op=n.operator.replace(/^aten\./,'').replace(/\.default$/,'');const m=moduleMap.get(n.module);return m?m.class_name+' · '+op:op;}
function transform(){world.setAttribute('transform',`translate(${tx},${ty}) scale(${scale})`);$('zoom-label').textContent=(scale*100<1?(scale*100).toFixed(2):Math.round(scale*100))+'%';}
function center(n,z=1){if(!n)return;scale=z;tx=vp.clientWidth/2-n.x*scale;ty=vp.clientHeight/2-n.y*scale;transform();}
function zoom(factor,x=vp.clientWidth/2,y=vp.clientHeight/2){const old=scale;scale=Math.min(2.5,Math.max(.0005,scale*factor));tx=x-(x-tx)*scale/old;ty=y-(y-ty)*scale/old;transform();}
function fit(){scale=Math.min(1,vp.clientWidth/view.width,vp.clientHeight/view.height)*.91;tx=(vp.clientWidth-view.width*scale)/2;ty=(vp.clientHeight-view.height*scale)/2;transform();}
function focusStructure(path){const parts=view.nodes.filter(n=>n.module===path||n.module.startsWith(path+'.'));if(!parts.length)return;const left=Math.min(...parts.map(n=>n.x-158)),right=Math.max(...parts.map(n=>n.x+158)),top=Math.min(...parts.map(n=>n.y-44)),bottom=Math.max(...parts.map(n=>n.y+44));scale=Math.min(1,(vp.clientWidth-50)/(right-left),(vp.clientHeight-65)/(bottom-top));tx=vp.clientWidth/2-(left+right)/2*scale;ty=(vp.clientHeight-25)/2-(top+bottom)/2*scale;transform();}
function initialize(){
 G=DATA.graphs[mode];nodes=new Map(G.nodes.map(n=>[n.id,n]));moduleMap=new Map(G.module_inventory.map(m=>[m.path,m]));owned=new Map();subtrees=new Map();groups=new Map();
 for(const m of G.module_inventory){owned.set(m.path,[]);subtrees.set(m.path,[]);}
 for(const n of G.nodes){if(owned.has(n.module))owned.get(n.module).push(n.id);for(const p of ancestors(n.module))if(subtrees.has(p))subtrees.get(p).push(n.id);}
 for(const m of G.module_inventory){const ids=subtrees.get(m.path);if(m.path!=='model'&&ids.length>1)groups.set('group:'+m.path,{id:'group:'+m.path,kind:'group',module:m.path,label:({backbone:'主干 · HGNetv2',encoder:'特征融合 · HybridEncoder',decoder:'解码与预测 · DFINETransformer'})[m.path]||m.path.split('.').pop()+' · '+m.class_name,members:ids});
  const direct=owned.get(m.path);if((direct.length>=8||m.path==='model'&&direct.length>1)&&m.children.length)groups.set('ops:'+m.path,{id:'ops:'+m.path,kind:'group',module:m.path,label:(m.path==='model'?'其他外部张量 · ':'本级张量操作 · ')+direct.length+'个',local:true,members:direct});
 }
 for(const g of groups.values()){
  const ids=new Set(g.members),boundary=G.edges.filter(e=>ids.has(e.source)&&!ids.has(e.target)),seen=new Set();g.outputs=[];
  for(const e of boundary){const key=e.source+':'+e.source_slot;if(seen.has(key))continue;seen.add(key);g.outputs.push({slot:key,shape:e.shape||[],dtype:nodes.get(e.source).outputs.find(o=>o.slot===e.source_slot)?.dtype||''});}
  g.width=316;g.height=88;
 }
 expanded=new Set();treeExpanded=new Set(['model']);selected=null;selectedModule=null;matches=[];matchIndex=0;renderModules();rebuild();fit();showEmpty();
}
function representative(n){
 for(const p of ancestors(n.module)){const key='group:'+p;if(groups.has(key)&&!expanded.has(key))return key;}
 const local='ops:'+n.module;if(groups.has(local)&&!expanded.has(local))return local;return n.id;
}
function buildView(){
 const key=mode+'|'+[...expanded].sort().join('|');if(cache.has(key))return cache.get(key);
 const mapping=new Map(G.nodes.map(n=>[n.id,representative(n)]));
 if(G.nodes.every(n=>mapping.get(n.id)===n.id)){const full={...G,representatives:mapping,internalEdges:0,full:true};cache.set(key,full);return full;}
 const visible=new Map();for(const n of G.nodes){const id=mapping.get(n.id);if(!visible.has(id))visible.set(id,{...(groups.get(id)||n)});}
 const edgeMap=new Map();let internalEdges=0;
 G.edges.forEach((e,i)=>{const source=mapping.get(e.source),target=mapping.get(e.target);if(source===target){internalEdges++;return;}const k=source+'>'+target+'|'+Boolean(e.math);if(!edgeMap.has(k))edgeMap.set(k,{...e,source,target,memberIndices:[]});edgeMap.get(k).memberIndices.push(i);});
 const edges=[...edgeMap.values()],graph=new dagre.graphlib.Graph({multigraph:true});graph.setGraph({rankdir:'TB',ranksep:65,nodesep:28,edgesep:12,ranker:'longest-path',marginx:35,marginy:35});graph.setDefaultEdgeLabel(()=>({}));
 for(const n of visible.values())graph.setNode(n.id,{width:316,height:88});edges.forEach((e,i)=>graph.setEdge(e.source,e.target,{},String(i)));dagre.layout(graph);
 for(const n of visible.values()){const p=graph.node(n.id);n.x=p.x;n.y=p.y;n.width=316;n.height=88;}
 edges.forEach((e,i)=>e.points=graph.edge({v:e.source,w:e.target,name:String(i)}).points);
 const result={nodes:[...visible.values()],edges,width:Math.ceil(graph.graph().width),height:Math.ceil(graph.graph().height),representatives:mapping,internalEdges,full:false};cache.set(key,result);if(cache.size>12)cache.delete(cache.keys().next().value);return result;
}
function drawGraph(target,v,interactive=true){
 target.replaceChildren();const elements=new Map(),edges=element('g',{'aria-label':'张量连接'});
 for(const e of v.edges){const d=e.points.map((p,i)=>(i?'L':'M')+p.x+','+p.y).join(' '),p=element('path',{d,class:'edge'+(e.math?' math':''),'marker-end':'url(#arrow)'});p.append(element('title',{},`${e.source} → ${e.target}\n${e.memberIndices?.length||1}条原始连接\n${shape(e.shape||[])}`));edges.append(p);}target.append(edges);
 for(const n of v.nodes){const group=element('g',{class:'node'+(n.kind==='math'?' math':'')+(n.kind==='group'?' group':''),'data-node':n.id,transform:`translate(${n.x-158},${n.y-44})`});
  group.append(element('rect',{width:316,height:88,rx:7,class:'frame'}),element('rect',{width:5,height:88,rx:2,class:'bar',fill:`var(--${branch(n)})`}),element('text',{x:14,y:21,class:'name'},short(name(n),43)),element('text',{x:14,y:40,class:'module-label'},short(n.module,45)));
  group.append(element('text',{x:14,y:62,class:'shape'},n.kind==='group'?n.members.length+'个内部节点 · '+n.outputs.length+'个边界张量':n.outputs[0]?shape(n.outputs[0].shape):'non-tensor output'));
  group.append(element('text',{x:14,y:79,class:'id'},n.kind==='group'?'▸ 已收起 · 点击右侧展开':n.id+(n.kind==='math'?' · 数学展开':n.kind==='module_identity'?' · 原生恒等调用':' · 实际算子')));
  group.append(element('title',{},n.module+'\n'+(n.kind==='group'?name(n)+'\n'+n.members.length+'个节点，数据完整保留':n.operator+'\n'+n.outputs.map(o=>o.slot+' '+shape(o.shape)).join('\n'))));
  if(interactive){group.setAttribute('role','button');group.setAttribute('tabindex','0');group.setAttribute('aria-label',name(n)+'，查看详情');group.addEventListener('click',()=>selectNode(n.id));group.addEventListener('keydown',e=>{if(e.target===group&&['Enter',' '].includes(e.key)){e.preventDefault();selectNode(n.id);}});
   if(n.kind==='group'){const action=element('foreignObject',{x:251,y:50,width:58,height:30,class:'node-action'}),b=document.createElement('button');b.textContent='展开';b.setAttribute('aria-label','展开 '+n.module+(n.local?' 本级操作':''));b.style.cssText='padding:4px 7px;font-size:12px;width:54px;height:26px';b.onclick=e=>{e.stopPropagation();toggleGroup(n.id,true);};action.append(b);group.append(action);}
  }
  target.append(group);elements.set(n.id,group);
 }return elements;
}
function rebuild(){view=buildView();representatives=view.representatives;nodeElements=drawGraph(world,view);if(selected&&nodeElements.has(selected))nodeElements.get(selected).classList.add('selected');highlightMatches();const folded=view.nodes.filter(n=>n.kind==='group').length;$('status').textContent=`${mode==='train'?'训练路径':'推理路径'} · 当前 ${view.nodes.length}个节点 / ${folded}个收起的结构 · 完整 ${G.nodes.length}个节点、${G.edges.length}条连接 · ${G.module_inventory.length}个注册模块全部保留。`;}
function renderModules(){
 treeRows=new Map();treeItems=new Map();$('module-total').textContent=G.module_inventory.length;const list=$('module-list');list.replaceChildren();
 function item(path){const m=moduleMap.get(path);if(!m)return null;const root=document.createElement('div');root.className='tree-item';root.setAttribute('role','treeitem');root.dataset.tree=path;root.setAttribute('aria-label',path);const row=document.createElement('div');row.className='tree-row';treeRows.set(path,row);treeItems.set(path,root);
  if(m.children.length){root.setAttribute('aria-expanded',treeExpanded.has(path));const toggle=document.createElement('button');toggle.className='tree-toggle';toggle.dataset.toggle=path;toggle.textContent=treeExpanded.has(path)?'▾':'▸';toggle.setAttribute('aria-label','折叠或展开 '+path);toggle.onclick=()=>{treeExpanded.has(path)?treeExpanded.delete(path):treeExpanded.add(path);updateTree(path);};row.append(toggle);}else{const blank=document.createElement('span');blank.className='tree-spacer';row.append(blank);}
  const b=document.createElement('button');b.className='module';b.dataset.module=path;b.title=path+' · '+m.class_name;b.innerHTML=`<span class="path stage-${path.split('.')[0]}"><span class="folder-icon">${m.children.length?'▱':'·'}</span><span class="basename">${esc(path.split('.').pop())}</span></span><span class="type">${esc(m.class_name)}${G.call_counts[path]?' · '+G.call_counts[path]+'次':''}</span>`;b.onclick=()=>selectModule(path);b.onkeydown=e=>{if(e.key==='ArrowRight'&&m.children.length){treeExpanded.add(path);updateTree(path);e.preventDefault();}if(e.key==='ArrowLeft'){if(treeExpanded.has(path)){treeExpanded.delete(path);updateTree(path);}else treeRows.get(parent(path))?.querySelector('.module').focus();e.preventDefault();}};row.append(b);root.append(row);
  if(m.children.length){const children=document.createElement('div');children.className='tree-children';children.setAttribute('role','group');children.hidden=!treeExpanded.has(path);for(const child of m.children){const nested=item(path==='model'?child:path+'.'+child);if(nested)children.append(nested);}root.append(children);}return root;
 }
 list.append(item('model'));
}
function updateTree(path){const item=treeItems.get(path);if(!item)return;const children=[...item.children].find(x=>x.classList.contains('tree-children'));if(children){children.hidden=!treeExpanded.has(path);item.setAttribute('aria-expanded',treeExpanded.has(path));treeRows.get(path).querySelector('.tree-toggle').textContent=treeExpanded.has(path)?'▾':'▸';}}
function markModule(path,reveal=false){for(const r of treeRows.values())r.classList.remove('selected');const row=treeRows.get(path);if(!row)return;row.classList.add('selected');selectedModule=path;if(reveal){for(const p of ancestors(path).slice(0,-1)){treeExpanded.add(p);updateTree(p);}row.scrollIntoView({block:'nearest',inline:'nearest'});}}
function sourceFor(m){while(m){if(m.source&&DATA.sources[m.source.file])return m;m=moduleMap.get(parent(m.path));}return null;}
function sourceButton(m){const s=sourceFor(m);return s?`<button id="view-source">${s===m?'查看原生源码':'父模块 '+esc(s.class_name)} · ${esc(s.source.file.split('/').pop())}:${s.source.line}</button>`:'';}
function wireSource(m){const b=$('view-source'),s=sourceFor(m);if(b&&s)b.onclick=()=>{const lines=DATA.sources[s.source.file].split('\n');$('source-title').textContent=s.class_name+' · '+s.source.file+' · class起始行 '+s.source.line;$('source-code').textContent=lines.map((l,i)=>String(i+1).padStart(4)+' '+l).join('\n');$('source-dialog').showModal();$('source-code').scrollTop=Math.max(0,(s.source.line-3)*17.6);};}
function actions(path,localKey){const key=localKey||'group:'+path;let html='';if(groups.has(key))html+=`<button id="toggle-structure" class="primary">${expanded.has(key)?'收起内部结构':'展开内部结构'}</button>`;const ancestor=ancestors(path).reverse().find(p=>p!==path&&p!=='model'&&groups.has('group:'+p)&&expanded.has('group:'+p));if(ancestor)html+=`<button id="collapse-parent">收起 ${esc(ancestor.split('.').pop())}</button>`;return html?`<div class="detail-actions">${html}</div>`:'';}
function wireActions(path,localKey){const b=$('toggle-structure');if(b)b.onclick=()=>toggleGroup(localKey||'group:'+path);const ancestor=ancestors(path).reverse().find(p=>p!==path&&p!=='model'&&groups.has('group:'+p)&&expanded.has('group:'+p));const c=$('collapse-parent');if(c&&ancestor)c.onclick=()=>toggleGroup('group:'+ancestor,false);}
function moduleDetails(m){if(!m)return '';return `<h2>${esc(m.class_name)}</h2><p class="path">${esc(m.path)}</p>${actions(m.path)}<p>直接参数：${m.direct_parameters.toLocaleString()}<br>含子模块参数：${m.parameters_total.toLocaleString()}<br>当前路径调用：${G.call_counts[m.path]||0}次<br>内部记录节点：${subtrees.get(m.path).length}个</p><pre>${esc(JSON.stringify(m.attributes,null,2))}</pre>${sourceButton(m)}`;}
function showEmpty(){$('details').innerHTML='<h2>节点信息</h2><p class="empty">点击大结构，再点击右侧“展开内部结构”。左侧目录箭头独立控制模块树的折叠。完整算子与连接均保留。</p>';}
function revealAncestors(path){let changed=false;for(const p of ancestors(path).slice(0,-1)){const key='group:'+p;if(groups.has(key)&&!expanded.has(key)){expanded.add(key);changed=true;}treeExpanded.add(p);updateTree(p);}return changed;}
function selectModule(path){const m=moduleMap.get(path);if(!m)return;if(revealAncestors(path))rebuild();markModule(path,true);const id='group:'+path,n=view.nodes.find(n=>n.id===id)||view.nodes.find(n=>n.module===path)||view.nodes.find(n=>n.module.startsWith(path+'.'));if(n){center(n);selected=n.id;for(const e of nodeElements.values())e.classList.remove('selected');nodeElements.get(n.id).classList.add('selected');}else selected=null;
 $('details').innerHTML=moduleDetails(m)+(subtrees.get(path).length?'':'<p class="note">当前路径没有独立forward记录；注册模块仍保留，参数可能通过父模块使用。</p>');wireSource(m);wireActions(path);
}
function showConnections(id){const incoming=view.edges.filter(e=>e.target===id),outgoing=view.edges.filter(e=>e.source===id),lookup=new Map(view.nodes.map(n=>[n.id,n]));const links=edges=>edges.map(e=>{const other=e.target===id?e.source:e.target,n=lookup.get(other);return `<button data-target="${esc(other)}">${esc(short(name(n),36))}<br>${e.memberIndices?.length||1}条连接 · ${esc(shape(e.shape||[]))}</button>`;}).join('');return `<h2 style="margin-top:18px">输入连接 · ${incoming.reduce((s,e)=>s+(e.memberIndices?.length||1),0)}</h2><div class="connections">${links(incoming)}</div><h2 style="margin-top:18px">输出连接 · ${outgoing.reduce((s,e)=>s+(e.memberIndices?.length||1),0)}</h2><div class="connections">${links(outgoing)}</div>`;}
function selectNode(id){let n=view.nodes.find(n=>n.id===id);if(!n&&nodes.has(id)){revealNode(id);n=view.nodes.find(n=>n.id===id);}if(!n)return;for(const e of nodeElements.values())e.classList.remove('selected');selected=id;nodeElements.get(id).classList.add('selected');const m=moduleMap.get(n.module);markModule(n.module,true);
 const rows=n.outputs.map(o=>o.slot+': '+shape(o.shape)+(o.dtype?' · '+o.dtype:'')).join('\n');
 $('details').innerHTML=`<h2>${esc(name(n))}</h2><p class="path">${esc(n.module)}${n.kind==='group'?'':'<br>'+esc(n.id+' · '+(n.native_operator||n.operator))}</p>${actions(n.module,n.local?n.id:undefined)}${n.kind==='group'?`<p>内部 ${n.members.length}个节点已收起，展开后逐层显示。</p><p class="note">下列尺寸是跨越此折叠结构的已记录连接。多条原始连接会在当前图中汇成一条线。</p>`:n.kind==='math'?'<p class="note">融合注意力内部的数学步骤；原生执行仍为一个融合算子。</p>':''}<pre>${esc(rows||(n.kind==='group'?'没有跨结构的输出连接记录':'无张量输出'))}</pre>${m?'<p>'+esc(m.class_name)+' · '+m.direct_parameters.toLocaleString()+'直接参数</p>':''}${sourceButton(m)}${showConnections(id)}${n.kind==='group'?'':'<h2 style="margin-top:18px">实际调用参数</h2><pre>'+esc(JSON.stringify(n.scalar_arguments,null,2))+'</pre>'}`;wireSource(m);wireActions(n.module,n.local?n.id:undefined);$('details').querySelectorAll('[data-target]').forEach(b=>b.onclick=()=>{const target=view.nodes.find(n=>n.id===b.dataset.target);center(target);selectNode(target.id);});
}
function toggleGroup(key,force){const g=groups.get(key);if(!g)return;const open=force??!expanded.has(key);revealAncestors(g.module);if(open){expanded.add(key);if(!g.local){treeExpanded.add(g.module);updateTree(g.module);}}else{expanded.delete(key);for(const k of [...expanded]){const x=groups.get(k);if(x&&(x.module.startsWith(g.module+'.')||(!g.local&&x.module===g.module)))expanded.delete(k);}if(!g.local){treeExpanded.delete(g.module);updateTree(g.module);}}
 rebuild();markModule(g.module,true);if(open){if(g.local){const n=view.nodes.find(n=>g.members.includes(n.id));center(n);selectNode(n.id);}else{selectModule(g.module);focusStructure(g.module);}}else{const n=view.nodes.find(n=>n.id===key);center(n);selectNode(key);}
}
function revealNode(id){const n=nodes.get(id);if(!n)return;let changed=revealAncestors(n.module);for(const k of ['group:'+n.module,'ops:'+n.module])if(groups.has(k)&&!expanded.has(k)){expanded.add(k);changed=true;}if(changed)rebuild();markModule(n.module,true);}
function highlightMatches(){for(const e of nodeElements.values())e.classList.remove('search-match');for(const n of matches)nodeElements.get(representatives.get(n.id))?.classList.add('search-match');}
function search(){const term=$('search').value.trim().toLowerCase();matches=term?G.nodes.filter(n=>(n.module+' '+n.operator+' '+n.id+' '+name(n)).toLowerCase().includes(term)):[];matchIndex=0;highlightMatches();$('matches').textContent=term?matches.length+'处':'';}
function next(){if(!matches.length)return;const index=matchIndex%matches.length,n=matches[index];revealNode(n.id);center(view.nodes.find(x=>x.id===n.id));selectNode(n.id);$('matches').textContent=`${index+1}/${matches.length}`;matchIndex++;}
function collapseAll(){expanded.clear();treeExpanded=new Set(['model']);for(const p of treeItems.keys())updateTree(p);selected=null;selectedModule=null;for(const r of treeRows.values())r.classList.remove('selected');rebuild();fit();showEmpty();}
function expandAll(){expanded=new Set(groups.keys());for(const m of G.module_inventory)if(m.children.length)treeExpanded.add(m.path);for(const p of treeItems.keys())updateTree(p);selected=null;rebuild();fit();showEmpty();}
function exportGraph(full){const v=full?G:view,copy=svg.cloneNode(false);copy.append(svg.querySelector('defs').cloneNode(true));const w=element('g',{id:'export-world'});copy.append(w);drawGraph(w,v,false);copy.setAttribute('xmlns',NS);copy.setAttribute('width',v.width);copy.setAttribute('height',v.height);copy.setAttribute('viewBox',`0 0 ${v.width} ${v.height}`);
 const style=document.createElementNS(NS,'style'),css=$('network-style-export')?.textContent||[...document.styleSheets].flatMap(s=>[...s.cssRules].map(r=>r.cssText)).join('\n');let resolved=css;const computed=getComputedStyle(document.documentElement);for(const token of ['bg','panel','ink','muted','line','backbone','encoder','decoder','gt','active'])resolved=resolved.replaceAll('var(--'+token+')',computed.getPropertyValue('--'+token).trim());style.textContent=resolved;copy.prepend(style);
 const blob=new Blob([new XMLSerializer().serializeToString(copy)],{type:'image/svg+xml'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`DEIM-D-FINE-S-${mode}-${full?'fully-expanded':'current-view'}.svg`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),10000);
}
$('mode').onchange=()=>{mode=$('mode').value;initialize();search();};$('search').oninput=search;$('search').onkeydown=e=>{if(e.key==='Enter')next();};$('next').onclick=next;$('jump').onchange=()=>selectModule($('jump').value);$('plus').onclick=()=>zoom(1.25);$('minus').onclick=()=>zoom(.8);$('start').onclick=()=>{const n=G.nodes.find(n=>n.kind==='input'&&n.operator.startsWith('RGB'));center(view.nodes.find(x=>x.id===representatives.get(n.id)));};$('fit').onclick=fit;$('expand-all').onclick=expandAll;$('collapse-all').onclick=collapseAll;$('export').onclick=()=>exportGraph(false);$('export-full').onclick=()=>exportGraph(true);$('inventory-toggle').onclick=()=>{$('workspace').classList.toggle('show-inventory');};$('close-source').onclick=()=>$('source-dialog').close();
let drag=null;vp.addEventListener('pointerdown',e=>{if(e.button!==0||e.target.closest('button'))return;drag={x:e.clientX,y:e.clientY,tx,ty,moved:false};});vp.addEventListener('pointermove',e=>{if(!drag)return;if(!drag.moved&&Math.hypot(e.clientX-drag.x,e.clientY-drag.y)>3){drag.moved=true;vp.setPointerCapture(e.pointerId);vp.classList.add('dragging');}if(drag.moved){tx=drag.tx+e.clientX-drag.x;ty=drag.ty+e.clientY-drag.y;transform();}});vp.addEventListener('pointerup',()=>{drag=null;vp.classList.remove('dragging');});vp.addEventListener('pointercancel',()=>{drag=null;vp.classList.remove('dragging');});vp.addEventListener('wheel',e=>{e.preventDefault();const r=vp.getBoundingClientRect();zoom(Math.exp(-e.deltaY*.0015),e.clientX-r.left,e.clientY-r.top);},{passive:false});
window.NETWORK_AUDIT={data:DATA,get mode(){return mode;},get selected(){return selected;},get displayedNodes(){return nodeElements.size;},get view(){return view;},get expanded(){return [...expanded];},get treeExpanded(){return [...treeExpanded];},get representatives(){return Object.fromEntries(representatives);},selectModule,selectNode,toggleGroup,collapseAll,expandAll};
initialize();
})();
