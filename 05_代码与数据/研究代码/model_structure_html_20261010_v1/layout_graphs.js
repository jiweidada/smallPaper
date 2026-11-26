const fs=require('fs'),path=require('path');
const dagre=require('./vendor/dagre.min.js');
const out=path.resolve(__dirname,'../../../outputs/model_structure_html_20261010_v1');
const data=JSON.parse(fs.readFileSync(path.join(out,'diagram_data.json'),'utf8'));
for(const [mode,graph] of Object.entries(data.graphs)){
  const g=new dagre.graphlib.Graph({multigraph:true});
  g.setGraph({rankdir:'LR',ranksep:66,nodesep:34,edgesep:10,ranker:'longest-path',marginx:50,marginy:50});g.setDefaultEdgeLabel(()=>({}));
  for(const n of graph.nodes)g.setNode(n.id,{width:316,height:88});
  graph.edges.forEach((e,i)=>g.setEdge(e.source,e.target,{},String(i)));
  dagre.layout(g);
  for(const n of graph.nodes){const p=g.node(n.id);n.x=Math.round(p.x*10)/10;n.y=Math.round(p.y*10)/10;n.width=316;n.height=88;}
  graph.edges.forEach((e,i)=>{e.points=g.edge({v:e.source,w:e.target,name:String(i)}).points.map(p=>({x:Math.round(p.x*10)/10,y:Math.round(p.y*10)/10}));});
  graph.width=Math.ceil(g.graph().width);graph.height=Math.ceil(g.graph().height);
  console.log(JSON.stringify({mode,nodes:graph.nodes.length,edges:graph.edges.length,width:graph.width,height:graph.height}));
}
fs.writeFileSync(path.join(out,'diagram_data_laid_out.json'),JSON.stringify(data));
require('./build_html.cjs')(data,path.join(out,'deim-dfine-s-full-network.html'));
