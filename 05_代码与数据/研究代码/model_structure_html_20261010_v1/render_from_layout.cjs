const fs=require('fs'),path=require('path');
const out=path.resolve(__dirname,'../../../outputs/model_structure_html_20261010_v1');
const data=JSON.parse(fs.readFileSync(path.join(out,'diagram_data_laid_out.json'),'utf8'));
require('./build_html.cjs')(data,path.join(out,'deim-dfine-s-full-network.html'));
console.log('HTML rendered from verified precomputed complete graph layouts');
