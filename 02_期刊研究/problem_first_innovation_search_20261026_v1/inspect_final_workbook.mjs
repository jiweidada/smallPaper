import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';
const out=path.dirname(fileURLToPath(import.meta.url));
const workbook=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(out,'373_问题优先研究_证据与文献对照.xlsx')));
const views=[
 ['问题与归因','E4:H7','preview_problem_explanations.png'],
 ['问题与归因','I4:K7','preview_problem_status.png'],
 ['文献与已有办法','G24:J26','preview_literature_limits.png']
];
for(const [sheetName,range,file] of views){
 const img=await workbook.render({sheetName,range,scale:1,format:'png'});
 await fs.writeFile(path.join(out,file),new Uint8Array(await img.arrayBuffer()));
}
console.log(JSON.stringify({reopenedSavedWorkbook:true,additionalRanges:views.length}));
