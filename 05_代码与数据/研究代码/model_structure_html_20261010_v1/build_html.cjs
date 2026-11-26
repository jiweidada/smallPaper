const fs=require('fs'),path=require('path');
module.exports=function build(data,target){
 const read=f=>fs.readFileSync(path.join(__dirname,f),'utf8');
 const safeScript=s=>s.replace(/<\//g,'<\\/');
 const html=read('network-template.html')
  .replace('/* ___NETWORK_STYLE___ */',()=>read('network-style.css'))
  .replace('___NETWORK_DATA___',()=>safeScript(JSON.stringify(data)))
  .replace('/* ___DAGRE_VENDOR___ */',()=>safeScript(read('vendor/dagre.min.js')))
  .replace('/* ___NETWORK_UI___ */',()=>safeScript(read('network-browser.js')));
 fs.writeFileSync(target,html);
};
