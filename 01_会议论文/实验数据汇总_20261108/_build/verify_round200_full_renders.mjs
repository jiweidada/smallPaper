import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const plan = JSON.parse(await fs.readFile(path.join(root, '_build/update_round200_plan.json'), 'utf8'));
const old = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(plan.snapshot, plan.xlsx_relative)));
const current = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root, plan.xlsx_relative)));
const results = [];
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
for (const [sheet, range, label] of [['终点对比', 'A1:F48', 'endpoints_full'], ['来源与权重', 'A1:C62', 'sources_full']]) {
  const hashes = [];
  for (const [wb, role] of [[old, 'before'], [current, 'after']]) {
    const blob = await wb.render({ sheetName: sheet, range, scale: 1.25, format: 'png' });
    const bytes = new Uint8Array(await blob.arrayBuffer());
    await fs.writeFile(path.join(root, `_build/previews/round200_${label}_${role}.png`), bytes);
    hashes.push(digest(bytes));
  }
  if (hashes[0] !== hashes[1]) throw new Error(`Protected full render changed: ${sheet}`);
  results.push({ sheet, range, before_sha256: hashes[0], after_sha256: hashes[1], identical: true });
}
const report = path.join(root, '_build/round200_workbook_checks.json');
const checks = JSON.parse(await fs.readFile(report, 'utf8'));
checks.full_protected_render_comparison = results;
await fs.writeFile(report, JSON.stringify(checks, null, 2) + '\n');
console.log(JSON.stringify({status:'FULL_PROTECTED_RENDERS_IDENTICAL', results}));
