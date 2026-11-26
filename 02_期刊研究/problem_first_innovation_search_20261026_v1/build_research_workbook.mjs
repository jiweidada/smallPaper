import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const out = path.dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(await fs.readFile(path.join(out,'research_tables_data.json'),'utf8'));
const workbook = Workbook.create();
const font = 'Microsoft YaHei';
const qa = [];
const column = n => { let s=''; for(let i=n+1;i>0;i=Math.floor((i-1)/26))s=String.fromCharCode(65+(i-1)%26)+s; return s; };
const textUnits = s => [...String(s??'')].reduce((a,c)=>a+(c.charCodeAt(0)>255?2:1),0);
const lines = (text,width) => String(text??'').split('\n').reduce((a,p)=>a+Math.max(1,Math.ceil(textUnits(p)/(width*1.05))),0);
for (let ti=0;ti<data.tables.length;ti++) {
  const t=data.tables[ti];
  const sheet=workbook.worksheets.add(t.name);
  const count=t.headers.length;
  const last=column(count-1);
  const lastRow=4+t.rows.length;
  sheet.showGridLines=false;
  sheet.tabColor=ti<2?'#294B70':'#7C8C9A';
  sheet.getRange(`A2:${last}2`).format.rowHeight=26;
  sheet.getRange('A2').values=[[t.title]];
  sheet.getRange('A2').format.font={name:font,size:14,bold:true,color:'#24364B'};
  const range=sheet.getRange(`A4:${last}${lastRow}`);
  range.values=[t.headers,...t.rows];
  range.format.font={name:font,size:11,color:'#24364B'};
  range.format.wrapText=true;
  range.format.verticalAlignment='top';
  range.format.fill='#FFFFFF';
  const tbl=sheet.tables.add(`A4:${last}${lastRow}`,true,`ResearchTable${ti+1}`);
  tbl.style='TableStyleMedium2';
  tbl.showFilterButton=true;
  t.widths.forEach((w,j)=>sheet.getRange(`${column(j)}4:${column(j)}${lastRow}`).format.columnWidth=w);
  const head=sheet.getRange(`A4:${last}4`);
  head.format.font={name:font,size:11,bold:true,color:'#FFFFFF'};
  head.format.fill='#294B70';
  head.format.horizontalAlignment='center';
  head.format.verticalAlignment='center';
  head.format.rowHeight=40;
  for(let r=0;r<t.rows.length;r++) {
    const demand=Math.max(...t.rows[r].map((v,j)=>lines(v,t.widths[j])));
    const height=Math.max(34,Math.ceil(demand*16+12));
    if(height>405)qa.push({sheet:t.name,row:r+5,heightDemand:height,issue:'row_needs_more_than_Excel_max_height'});
    sheet.getRange(`A${r+5}:${last}${r+5}`).format.rowHeight=Math.min(405,height);
  }
  sheet.getRange(`A5:B${lastRow}`).format.font={name:font,size:11,bold:true,color:'#24364B'};
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(ti<2?2:1);
}
workbook.recalculate();
const inspections=[];
for(const t of data.tables) {
  inspections.push({sheet:t.name,expectedDataRows:t.rows.length,inspection:(await workbook.inspect({kind:'table',range:`'${t.name}'!A4:C7`,include:'values,formulas',tableMaxRows:4,tableMaxCols:3,tableMaxCellChars:200,maxChars:2500})).ndjson});
}
const errors=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},summary:'Final formula error scan',maxChars:3000});
await fs.writeFile(path.join(out,'workbook_inspection.json'),JSON.stringify({qa,inspections,errors:errors.ndjson},null,2));
if(qa.length) throw new Error(`Required text may not fit Excel row heights: ${JSON.stringify(qa)}`);
const filename='373_问题优先研究_证据与文献对照.xlsx';
const xlsx=await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(out,filename));
for(let i=0;i<data.tables.length;i++) {
  const t=data.tables[i];
  const preview=await workbook.render({sheetName:t.name,range:'A2:D6',scale:1.25,format:'png'});
  await fs.writeFile(path.join(out,`preview_${i+1}.png`),new Uint8Array(await preview.arrayBuffer()));
}
await fs.writeFile(path.join(out,'workbook_build_receipt.json'),JSON.stringify({file:path.join(out,filename),sheets:data.tables.map(t=>({name:t.name,dataRows:t.rows.length,columns:t.headers.length})),recalculated:true,previewedAllSheets:true,qa},null,2));
console.log(JSON.stringify({file:filename,sheets:data.tables.length,rows:data.tables.map(t=>t.rows.length),previews:5}));
