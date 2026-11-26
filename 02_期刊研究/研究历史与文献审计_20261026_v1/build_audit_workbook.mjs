import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const auditDir=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(auditDir,'../..');
const outDir=path.join(root,'outputs','research_history_literature_audit_20261026_v1');
await fs.mkdir(outDir,{recursive:true});
const read=async n=>JSON.parse(await fs.readFile(path.join(auditDir,n),'utf8'));
const dirs=await read('direction_table_data.json');
const papers=await read('literature_table_data.json');
const sum=await read('assembled_summary.json');
const label='372_研究方向与文献覆盖审计_20261026_v1';
const cite=e=>e?`${e.path}:${e.line}`:'';
const yn=v=>v?'是':'否';
const col=n=>{let r='';while(n>0){n--;r=String.fromCharCode(65+n%26)+r;n=Math.floor(n/26)}return r};
const cut=(s,n=430)=>s.length>n?s.slice(0,n)+'…（余文见原记录）':s;
const rank=s=>/§|第\s*\d+\s*[页节]|(?:Eq|式)\s*\(?\d|方法.*(?:核读|局部|复读|补读)|源码/.test(s)?2:/摘要|引言|abstract|introduction/i.test(s)?1:0;
function bestScopes(p){
  const rows=[...p.historical_reading_evidence].sort((a,b)=>rank(b.text)-rank(a.text));
  const uniq=[];const seen=new Set();
  for(const x of rows){if(x.text&&!seen.has(x.text)){seen.add(x.text);uniq.push(x)}}
  return uniq.slice(0,2);
}
const dHeader=['条目ID','问题大类','方案或诊断假设','当前主状态','实际验证层级','实际结果及比较口径','停止或保留原因','证据否定范围与归因边界','所解决的问题','机制假设','别名与合并范围','研究单元类型','当前阶段检查完成','有计划训练终点记录','已确认用途的文献ID','主证据路径与位置','补充证据路径','证据来源层级','去重及统计口径'];
const dData=dirs.map(d=>[
  d.id,`${d.family} ${d.family_name}`,d.name,d.status,d.stage,d.result,d.stop_reason,d.negative_scope,d.problem,d.hypothesis,d.aliases,d.unit_type,
  yn(d.current_check_complete),yn(d.has_planned_training_endpoint),
  papers.filter(p=>p.actual_related_direction_ids.includes(d.id)).map(p=>p.id).join('、')||'本轮未逐篇对齐；不是没有先例',
  cite(d.evidence),d.secondary_evidence.map(cite).join('\n')||'未另列',d.evidence_level,d.count_note
]);
const lHeader=['文献ID','实际题名或原记录简称','单一主主题','本地资料状态','题名与身份核实情况','历史阅读证据等级','在项目中的已记录用途与边界','主题关联的问题类','实际任务范围','历史阅读原记录路径与位置','历史阅读范围节选','公式或代码对照记录','全文阅读情况','本轮实际核查范围','PDF路径与版本','保存的原始来源URL','稳定标识与版本归并','本地PDF SHA256','主题归类依据','未决项与补充来源'];
const lData=papers.map(p=>{
  const scopes=bestScopes(p);
  const uses=p.documented_project_uses.map(u=>`${u.direction_ids.join('、')}：${u.use_and_limit}\n来源：${cite(u.evidence)}`).join('\n\n');
  return [p.id,p.title,p.theme_name,p.availability,p.title_verification,p.historical_reading_grade,
    uses||'本轮未逐项对齐实际用途。不能推断未使用；主题关联不代表采用或排除。',
    p.related_problem_families.join('、')||'主题待核',p.scope,
    scopes.map(s=>`${s.source}；${s.locator}`).join('\n')||'未对齐到可核阅读范围的历史记录',
    scopes.map(s=>cut(s.text)).join('\n\n')||'未知；不把本轮文本提取算历史阅读',
    p.has_formula_or_code_comparison_record?'有可定位对照记录；见阅读来源':'本轮未匹配明确对照记录；不表示未做过',
    '无法确认；局部阅读不升级为全文已读',p.current_audit_scope,
    p.pdf_files.join('\n')||'无本地主文PDF',p.urls.join('\n')||'本地PDF/原索引为依据；未恢复来源URL',
    p.stable_keys.join('\n')||'未取得足够的稳定标识',p.pdf_sha256.join('\n')||'不适用：无PDF',p.theme_basis,
    `阅读范围为最明确的至多两条节选，完整已存证据见：${path.join(auditDir,'literature_table_data.json')}（${p.id}）。\n${p.use_coverage_note}`
  ];
});
for(const [name,data] of [['directions',dData],['literature',lData]]){
  for(const [i,row] of data.entries())for(const [j,value] of row.entries()){
    if(typeof value!=='string')throw new Error(`${name} ${i}/${j} is not text`);
    if(value.length>32767)throw new Error(`${name} ${i}/${j} exceeds Excel cell limit`);
  }
}
const wb=Workbook.create();
function buildSheet(name,headers,data,widths,tableName){
  const sh=wb.worksheets.add(name);const last=col(headers.length);const bottom=data.length+1;
  sh.showGridLines=false;
  sh.getRange(`A1:${last}${bottom}`).values=[headers,...data];
  const all=sh.getRange(`A1:${last}${bottom}`);
  all.format.font={name:'Arial',size:10,color:'#243247'};
  all.format.wrapText=true;
  all.format.verticalAlignment='top';
  all.setNumberFormat('@');
  for(let c=0;c<widths.length;c++)sh.getRange(`${col(c+1)}1:${col(c+1)}${bottom}`).format.columnWidthPx=widths[c];
  const table=sh.tables.add(`A1:${last}${bottom}`,true,tableName);
  table.showFilterButton=true;
  sh.getRange(`A2:${last}${bottom}`).format.fill='#FFFFFF';
  sh.getRange(`A1:${last}1`).format={fill:'#263D55',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,verticalAlignment:'center',horizontalAlignment:'center',rowHeightPx:44};
  sh.getRange(`A1:${last}1`).format.borders={preset:'inside',style:'thin',color:'#FFFFFF'};
  sh.getRange(`A2:A${bottom}`).format.font={name:'Arial',size:10,bold:true,color:'#263D55'};
  let largestHeight=0;
  for(let r=0;r<data.length;r++){
    let lineCount=1;
    for(let c=0;c<data[r].length;c++){
      const charsPerLine=Math.max(7,(widths[c]-12)/7.5);
      const lines=data[r][c].split('\n').reduce((n,s)=>n+Math.max(1,Math.ceil([...s].reduce((v,ch)=>v+(ch.codePointAt(0)>255?1.8:1),0)/charsPerLine)),0);
      lineCount=Math.max(lineCount,lines);
    }
    const height=Math.max(92,Math.ceil(lineCount*16+14));
    if(height>540)throw new Error(`Review overly long visible row ${name}:${r+2}, height ${height}`);
    sh.getRange(`A${r+2}:${last}${r+2}`).format.rowHeightPx=height;
    largestHeight=Math.max(largestHeight,height);
    if(r%2===1)sh.getRange(`A${r+2}:${last}${r+2}`).format.fill='#F4F7FA';
  }
  sh.freezePanes.freezeRows(1);
  sh.freezePanes.freezeColumns(3);
  console.log(JSON.stringify({sheet:name,records:data.length,columns:headers.length,largestHeight}));
  return sh;
}
const ds=buildSheet('探索方向',dHeader,dData,[68,170,265,126,210,370,300,340,230,255,290,110,102,130,220,470,470,270,290],'ResearchDirections');
const ls=buildSheet('文献主题与阅读证据',lHeader,lData,[68,430,210,170,240,190,430,120,280,500,500,230,225,230,520,460,360,500,280,500],'LiteratureEvidence');
const colors={'正面证据':'#E3EFE9','混合或受限':'#FFF2D5','训练未获收益':'#F8E5E5','预检未过':'#F8E5E5','实现或协议无效':'#EDE4F3','未完成或未验证':'#F0F1F3','与简单对照或先例重叠':'#E8ECF6','诊断事实/无方法验收':'#E8F0F5'};
for(const [i,d] of dirs.entries())ds.getRange(`D${i+2}`).format.fill=colors[d.status];
for(const [i,p] of papers.entries())if(p.title_verification.includes('待核'))ls.getRange(`E${i+2}`).format.fill='#FFF2D5';
wb.recalculate();
const inspections={};
for(const sheet of ['探索方向','文献主题与阅读证据'])inspections[sheet]=(await wb.inspect({kind:'region',sheetId:sheet,range:'A1:F4',maxChars:2400,tableMaxCellChars:100})).ndjson;
inspections.errors=(await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},maxChars:1800})).ndjson;
await fs.writeFile(path.join(auditDir,'workbook_inspections.json'),JSON.stringify(inspections,null,2),'utf8');
for(const [sheet,name] of [['探索方向','directions_preview.png'],['文献主题与阅读证据','literature_preview.png']]){
  const blob=await wb.render({sheetName:sheet,range:'A1:F5',scale:1,format:'png'});
  await fs.writeFile(path.join(auditDir,name),new Uint8Array(await blob.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(outDir,label+'_两张明细表.xlsx'));

const report=`研究方向与文献覆盖审计　截至2026-10-26

一、探索了多少，为什么没有走通
按“问题—机制—方案”归并改名、修订、seed和对照臂，现存可追溯记录共10个问题大类、73个研究条目：60个方法/干预方案，13个诊断假设。73不是73个独立创新点。57条完成各自当前阶段检查；15条有计划训练预算的终点记录，不能称15条充分训练或公平验证均完成。
主状态逐条互斥：正面证据2、混合/受限12、预检未过33、训练未获收益4、与简单对照/先例重叠4、协议/实现无效1、未完成/未验证13、诊断事实且无方法验收4，共73。“多数失败”不能用于概括所有停止；这些状态也不能用作统一AP成功率。
反复出现的障碍是：同一浅层细节、几何读取、匹配/候选问题换接口或名字继续尝试；几何可达、oracle救回、局部损失等代理没有转为最终AP收益；同seed/预算/初始化的强简单对照削弱了机制归因。另有早停和协议错误：support-matching只到4/5轮、因任务转向停止（D035）；SCAP先验缓存不一致（D041）。它们不能计成已完成训练失败。成熟96轮的C机制证据与12轮Mosaic正结果各有限定范围（D007、D048）；大量近期筛查基于BASE24，不能直接推出成熟模型的统一病因。

二、文献有多少，实际覆盖什么
主库260个PDF（根258、子目录2），SHA去重255份，再归并同文版本和补充材料，得到252个文献身份：251有主文、1仅补充材料。另恢复60条无本地主文的检索/引用记录，表共312行，其中17行完整题名仍待核；312不是312篇已核实独立论文，历史总检索篇数无法完整恢复。
252个PDF文献按单一主主题分组：分辨率/裁块/多尺度53、几何/回归/分布39、query/匹配/监督38、细节/下采样/频率36、数学统计/迁移25、类别/评分/校准23、通用架构/效率15、增强/预训练12、数据/标注/评估6、时序/视频5。前述分辨率、几何、query、细节、评分五类合计189/252=75%，与项目反复探索的机制高度集中。跨任务统计、分割、定位论文已单列任务边界，不能全当同任务直接先例。
PDF文献中，可定位的历史记录下限为局部方法/公式76、摘要/引言18、未知158；明确公式/代码对照记录匹配到7。未确认全文已读篇数。未知不等于没读；本轮首1–2页核对不算历史精读。已逐项列出34个记录的实际用途及边界，其余用途未逐篇对齐，不能据此声称大量文献未使用。
“提出候选后补查近邻”有可证实实例（C/355与SPD、CFCI/ESE；集合监督与TOLF），但无法给出全历史发生比例。把NWD/SA-Matching的存在当成“问题已解决”有明确越界：本地未做其同协议比较。项目尚未形成稳定的“问题证据—竞争解释—方法增量”闭环；现有材料也没有证明其它问题或方法空间已被穷尽。

证据位置：两表各行均列原文件、行号/记录键及归因边界。重点原始来源：
[D035] ${cite(dirs.find(d=>d.id==='D035').evidence)}
[D041] ${cite(dirs.find(d=>d.id==='D041').evidence)}
[D048] ${path.join(root,'outputs/scale_protection12_20261010_v1/fixed12_policy_decision_v1.json')}（seeds）
[先例闸门勘误] ${path.join(root,'英文小论文/持续寻找研究主线_20261020_v2/360_v7_问题定义交付_独立核对_20261026.txt')}:57

范围：核查项目入口、交接/历史复盘、实验终点和执行记录、代码/配置、文献首面及阅读/来源账本，覆盖早期方案至370系列。不是逐字审完所有原始会话；总检索与全部想法的绝对总数不可恢复。150处长路径目录入口元数据已补读，20处依赖包目录仍拒绝访问，不影响已列源文件核对。完整计数、哈希、去重与阅读证据保存在 ${auditDir}。本轮未跑实验、模型前向、训练或新联网检索，未改旧文件；交付后停止。
`;
await fs.writeFile(path.join(outDir,label+'_一页结论.txt'),report,'utf8');
await fs.writeFile(path.join(auditDir,'delivery_manifest.json'),JSON.stringify({created_20261026:true,workbook:path.join(outDir,label+'_两张明细表.xlsx'),conclusion:path.join(outDir,label+'_一页结论.txt'),direction_records:dirs.length,literature_records:papers.length,summary:sum,report_characters:report.length},null,2),'utf8');
console.log(JSON.stringify({outputDir:outDir,directionRecords:dirs.length,literatureRecords:papers.length,reportCharacters:report.length}));
