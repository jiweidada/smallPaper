import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const planArg = process.argv.indexOf("--plan");
const planName = planArg < 0 ? "update_round30_plan.json" : process.argv[planArg + 1];
const plan = JSON.parse(await fs.readFile(path.join(root, "_build", planName), "utf8"));
const tag = plan.update_tag ?? "round30";
const rounds = plan.target_rounds ?? 30;
const localDate = plan.local_date;
const summary = JSON.parse(await fs.readFile(path.join(root, "实验汇总.json"), "utf8"));
const file = path.join(root, plan.xlsx_relative);
const previewDir = path.join(root, "_build/previews");
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
const progress = workbook.worksheets.getItem("NO_MSCA进度");
const protectedRanges = [["终点对比", "A1:F48"], ["来源与权重", "A1:C62"]];
const hash = bytes => createHash("sha256").update(bytes).digest("hex");

async function render(wb, name, range, filename) {
  const blob = await wb.render({ sheetName: name, range, scale: 1.25, format: "png" });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  await fs.writeFile(path.join(previewDir, filename), bytes);
  return hash(bytes);
}

function valuesAndFormulas(wb) {
  return Object.fromEntries(protectedRanges.map(([name, range]) => {
    const cells = wb.worksheets.getItem(name).getRange(range);
    return [name, { values: cells.values, formulas: cells.formulas }];
  }));
}

if (process.argv.includes("--preview-only")) {
  if (hash(await fs.readFile(file)) !== plan.old_xlsx_sha256) throw new Error("Pre-edit workbook changed");
  const before = {
    protected: valuesAndFormulas(workbook),
    sheetNames: [0, 1, 2].map(index => workbook.worksheets.getItemAt(index).name),
    rendered: {
      "终点对比": await render(workbook, "终点对比", "A1:F45", `${tag}_endpoints_before.png`),
      "来源与权重": await render(workbook, "来源与权重", "A1:C50", `${tag}_sources_before.png`),
    },
  };
  await render(workbook, "NO_MSCA进度", "A1:D33", `${tag}_progress_before.png`);
  await fs.writeFile(path.join(root, `_build/${tag}_workbook_before.json`), JSON.stringify(before));
  console.log(JSON.stringify({ status: "PRE_EDIT_WORKBOOK_RENDERED", worksheet: "NO_MSCA进度" }));
  process.exit(0);
}

const before = JSON.parse(await fs.readFile(path.join(root, `_build/${tag}_workbook_before.json`), "utf8"));
if (JSON.stringify(valuesAndFormulas(workbook)) !== JSON.stringify(before.protected)) {
  throw new Error("Protected workbook content changed before editing");
}
progress.getRange("A3").values = [[`${localDate}，30715续训运行；已核验${rounds}轮，现场第166轮。320轮终点空缺。`]];
progress.getRange("B5").values = [[`第${rounds}轮进度`]];
progress.getRange("B6:B17").values = summary.metric_order.map(metric => [plan.new_NO_MSCA_metrics[metric]]);
progress.getRange("B20").values = [[rounds]];
progress.getRange("B22").values = [[plan.total_updates ?? summary.NO_MSCA_progress.optimizer_step_attempts]];
progress.getRange("A28:A33").values = plan.progress_notes ? plan.progress_notes.map(line => [line]) : [
  ["第29轮后换机，原始模型、EMA、优化器、warmup及保存RNG恢复已核验。"],
  ["新增1轮和1617次更新已核验，累计30轮。"],
  ["来源 NO_MSCA_RECOVERED_R30 和 NO_MSCA_NATIVE_R30。"],
  ["NO_MSCA_LIVE_05另存第31轮运行快照，指标截至第30轮。"],
  ["失联后未保留的工作量未知，完整物理计算量无法确认。"],
  ["恢复子合约的live=false仅说明该子报告不做远程实时检查。"],
];
workbook.recalculate();
if (JSON.stringify(valuesAndFormulas(workbook)) !== JSON.stringify(before.protected)) {
  throw new Error("Protected completed tables or sources changed during the targeted edit");
}
if (progress.getRange("C6:D17").values.flat().some(value => value !== null && value !== "")) {
  throw new Error("NO_MSCA terminal metrics must remain blank");
}
const errors = await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 20 }, maxChars: 1500,
});
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(file);
const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
const after = valuesAndFormulas(saved);
if (JSON.stringify(after) !== JSON.stringify(before.protected)) {
  throw new Error("Protected completed tables or sources changed in the saved workbook");
}
const names = [0, 1, 2].map(index => saved.worksheets.getItemAt(index).name);
if (JSON.stringify(names) !== JSON.stringify(before.sheetNames)) throw new Error("Worksheet order changed");
const savedProgress = saved.worksheets.getItem("NO_MSCA进度");
summary.metric_order.forEach((metric, index) => {
  const actual = savedProgress.getRange(`B${index + 6}`).values[0][0];
  if (Math.abs(actual - plan.new_NO_MSCA_metrics[metric]) > 1e-15) throw new Error("Saved progress value changed");
});
if (savedProgress.getRange("C6:D17").values.flat().some(value => value !== null && value !== "")) {
  throw new Error("Saved terminal cells must remain blank");
}
const rendered = {
  "终点对比": await render(saved, "终点对比", "A1:F45", `${tag}_endpoints_after.png`),
  "来源与权重": await render(saved, "来源与权重", "A1:C50", `${tag}_sources_after.png`),
};
if (JSON.stringify(rendered) !== JSON.stringify(before.rendered)) {
  throw new Error("A protected worksheet visual changed after export");
}
await render(saved, "NO_MSCA进度", "A1:D33", `${tag}_progress_after.png`);
const inspection = await saved.inspect({
  kind: "table", range: "NO_MSCA进度!A5:D25", include: "values,formulas",
  tableMaxRows: 21, tableMaxCols: 4, maxChars: 3500,
});
await fs.writeFile(path.join(root, `_build/${tag}_workbook_inspection.ndjson`), inspection.ndjson + "\n" + errors.ndjson);
await fs.writeFile(path.join(root, `_build/${tag}_workbook_checks.json`), JSON.stringify({
  status: "PROGRESS_ONLY_EDIT_EXPORTED_AND_VERIFIED", local_date: localDate, NO_MSCA_accepted_rounds: rounds,
  protected_values_formulas_and_sources_identical: true, protected_sheet_renders_identical: true,
  NO_MSCA_metric_values_checked: 12, final_metric_cells_blank: 24, worksheet_names_unchanged: true,
  error_scan: errors.ndjson, xlsx_sha256: hash(await fs.readFile(file)),
}, null, 2) + "\n");
console.log(JSON.stringify({
  status: "PROGRESS_ONLY_EDIT_VERIFIED", output: file, xlsx_sha256: hash(await fs.readFile(file)),
}));
