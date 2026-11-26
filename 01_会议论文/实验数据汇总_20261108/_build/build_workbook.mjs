import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const buildDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(buildDir);
const data = JSON.parse(await fs.readFile(path.join(root, "实验汇总.json"), "utf8"));
const audit = JSON.parse(await fs.readFile(path.join(root, "输入核查.json"), "utf8"));
const outputDir = path.join(root, "outputs", "实验数据汇总_20261108");
const previewDir = path.join(buildDir, "previews");
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const workbook = Workbook.create();
const result = workbook.worksheets.add("终点对比");
const progress = workbook.worksheets.add("NO_MSCA进度");
const sources = workbook.worksheets.add("来源与权重");
const byEndpoint = new Map(data.completed_endpoints.map(row => [row.arm + "/" + row.endpoint_role, row]));
const arms = ["BASE", "FULL", "NO_MPS"];
const bodyFont = "Arial";
const headerFill = "#243B53";
const textColor = "#172B4D";

function styleSheet(sheet, range) {
  sheet.showGridLines = false;
  sheet.getRange(range).format.font = { name: bodyFont, size: 11, color: textColor };
  sheet.getRange(range).format.verticalAlignment = "center";
  sheet.getRange(range).format.rowHeight = 23;
  sheet.getRange(range).format.wrapText = false;
}

function title(sheet, cell, text) {
  sheet.getRange(cell).values = [[text]];
  sheet.getRange(cell).format.font = { name: bodyFont, size: 15, bold: true, color: textColor };
  sheet.getRange(cell).format.rowHeight = 30;
}

function header(sheet, range, values) {
  sheet.getRange(range).values = [values];
  sheet.getRange(range).format = {
    fill: headerFill,
    font: { name: bodyFont, size: 11, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    rowHeight: 39,
    wrapText: true,
  };
}

function metricBlock(start, role) {
  const rounds = arms.map(arm => byEndpoint.get(arm + "/" + role).round_1based);
  header(result, `A${start}:F${start}`,
    ["指标", ...arms.map((arm, index) => arm + " 第" + rounds[index] + "轮"), "FULL−BASE\n（百分点）", "FULL−NO_MPS\n（百分点）"]);
  const rows = data.metric_order.map(metric => [metric, ...arms.map(arm => byEndpoint.get(arm + "/" + role).metrics_fraction[metric]), null, null]);
  result.getRange(`A${start + 1}:F${start + 12}`).values = rows;
  result.getRange(`B${start + 1}:D${start + 12}`).setNumberFormat("0.0000%");
  result.getRange(`B${start + 1}:F${start + 12}`).format.horizontalAlignment = "right";
  result.getRange(`E${start + 1}:F${start + 12}`).formulas = data.metric_order.map((_, index) => {
    const row = start + index + 1;
    return [`=(C${row}-B${row})*100`, `=(C${row}-D${row})*100`];
  });
  const deltas = result.getRange(`E${start + 1}:F${start + 12}`);
  deltas.setNumberFormat("+0.0000;-0.0000;0.0000");
  deltas.conditionalFormats.add("cellIs", {
    operator: "lessThan", formula: 0, format: { font: { color: "#9C2D2D" } },
  });
}

styleSheet(result, "A1:F48");
result.tabColor = "#243B53";
result.getRange("A1:A48").format.columnWidth = 17;
result.getRange("B1:D48").format.columnWidth = 21;
result.getRange("E1:F48").format.columnWidth = 27;
title(result, "A2", "固定320轮主端点");
result.getRange("A3").values = [["EMA，晚期回滚前的实际评估状态。指标为百分比，差值为百分点。"]];
result.getRange("A3").format.font.italic = true;
metricBlock(5, "primary");
title(result, "A19", "AP最佳次端点");
result.getRange("A20").values = [["同一320轮轨迹的最高AP单轮，保留该轮全部12项指标。"]];
result.getRange("A20").format.font.italic = true;
metricBlock(22, "secondary");
header(result, "A36:F36", ["配置", "训练参数量", "主端点轮次", "次端点轮次", "seed", "来源"]);
result.getRange("A37:F39").values = arms.map(arm => {
  const primary = byEndpoint.get(arm + "/primary");
  return [arm, primary.trainable_parameters, 320, byEndpoint.get(arm + "/secondary").round_1based, 0, primary.source_id];
});
result.getRange("B37:E39").setNumberFormat("#,##0");
result.getRange("E37:E39").format.horizontalAlignment = "center";
const methodLines = [
  "本地转换COCO bbox，非官方VisDrone。548张验证图，640×640，maxDets 1/10/100，top-300。",
  "BASE在第214和308轮后续训，FULL在第1轮后续训，原始锚点缺少主进程RNG。",
  "FULL首轮逐批学习率日志缺失。NO_MPS从索引0开始，轨迹与容量差异均保留。",
  "单seed 0，AP最佳由同一验证集选取。NO_MSCA仅列已核验进度。",
];
result.getRange("A42:A45").values = methodLines.map(line => [line]);
result.getRange("A42:A45").format.font = { name: bodyFont, size: 11, color: "#52657A" };

styleSheet(progress, "A1:D33");
progress.getRange("A1:A33").format.columnWidth = 24;
progress.getRange("B1:B33").format.columnWidth = 23;
progress.getRange("C1:D33").format.columnWidth = 30;
title(progress, "A2", "NO_MSCA 已核验进度");
progress.getRange("A3").values = [["第29轮来源快照。320轮终点保持空缺。"]];
header(progress, "A5:D5", ["指标", "第29轮进度", "第320轮主端点", "320轮AP最佳次端点"]);
progress.getRange("A6:D17").values = data.metric_order.map(metric =>
  [metric, data.NO_MSCA_progress.latest_metrics_fraction[metric], null, null]);
progress.getRange("B6:D17").setNumberFormat("0.0000%");
progress.getRange("B6:D17").format.horizontalAlignment = "right";
progress.getRange("A20:B25").values = [
  ["已核验轮数", 29], ["目标轮数", 320], ["已核验更新尝试", 46893], ["训练参数量", 3699662],
  ["seed", 0], ["终点状态", "空缺"],
];
progress.getRange("B20:B24").setNumberFormat("#,##0");
progress.getRange("A28").values = [["来源为NO_MSCA_ROUND29与NO_MSCA_LINEAGE，仅作进度记录。"]];
progress.getRange("A29").values = [["完整rolling及EMA哈希保留于实验汇总JSON。"]];

styleSheet(sources, "A1:C62");
sources.getRange("A1:A62").format.columnWidth = 28;
sources.getRange("B1:B62").format.columnWidth = 80;
sources.getRange("C1:C62").format.columnWidth = 77;
title(sources, "A2", "输入来源与SHA-256");
sources.getRange("A3").values = [[`${audit.check_count}项核查通过，29个本地证据文件哈希一致。`]];
header(sources, "A5:C5", ["来源ID", "相对远程公平对照目录的路径", "文件 SHA-256"]);
sources.getRange("A6:C14").values = data.input_sources.map(source =>
  [source.source_id, source.relative_to_remote_root, source.sha256]);
sources.getRange("B6:B14").format.wrapText = true;
sources.getRange("A6:C14").format.rowHeight = 43;
title(sources, "A17", "端点权重与配方绑定");
header(sources, "A19:C19", ["配置和端点", "字段", "SHA-256"]);
const bindings = [];
for (const record of data.completed_endpoints) {
  const label = record.arm + " " + (record.endpoint_role === "primary" ? "主端点" : "次端点") + " r" + record.round_1based;
  for (const [key, display] of [
    ["checkpoint_sha256", "checkpoint文件"], ["ema_tensor_sha256", "实际评估EMA张量"],
    ["protocol_sha256", "冻结协议"], ["configuration_sha256", "配置"],
  ]) {
    bindings.push([label, display, record[key]]);
  }
}
sources.getRange(`A20:C${19 + bindings.length}`).values = bindings;
sources.getRange("A47").values = [["口径与限制"]];
sources.getRange("A47").format.font.bold = true;
sources.getRange("A48").values = [["完整预测、数组、指标SHA和输入绝对路径见JSON与CSV。"]];
sources.getRange("A49").values = [["AP最佳为验证集选点。所有配置为单seed，未进行新统计推断。"]];
sources.getRange("A50").values = [["子报告full_pair_complete保留原值，总报告及320轮核验为本次收录依据。"]];
sources.freezePanes.freezeRows(5);

// Perturb one numerical input, verify both dependent differences, then restore it.
const original = result.getRange("C6").values[0][0];
const baseline = result.getRange("E6:F6").values[0];
result.getRange("C6").values = [[original + 0.001]];
const perturbed = result.getRange("E6:F6").values[0];
if (!perturbed.every((value, index) => Math.abs(value - baseline[index] - 0.1) < 1e-10)) {
  throw new Error("Percentage-point formulas failed the input-change check");
}
result.getRange("C6").values = [[original]];
workbook.recalculate();
let formulaChecks = 0;
for (const [role, start] of [["primary", 5], ["secondary", 22]]) {
  const actual = result.getRange(`E${start + 1}:F${start + 12}`).values;
  data.metric_order.forEach((metric, index) => {
    ["FULL-BASE", "FULL-NO_MPS"].forEach((comparison, column) => {
      const expected = data.differences.find(row => row.endpoint_role === role && row.metric === metric &&
        row.comparison === comparison).difference_percentage_points;
      if (Math.abs(actual[index][column] - expected) > 1e-12) throw new Error("Workbook difference mismatch");
      formulaChecks += 1;
    });
  });
}
if (progress.getRange("C6:D17").values.flat().some(value => value !== null && value !== "")) {
  throw new Error("NO_MSCA final endpoints must stay blank");
}
const inspection = await workbook.inspect({
  kind: "table", range: "终点对比!A5:F17", include: "values,formulas", tableMaxRows: 13, tableMaxCols: 6, maxChars: 4500,
});
const errorScan = await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 30 }, maxChars: 2000,
});
await fs.writeFile(path.join(buildDir, "workbook_inspection.ndjson"), inspection.ndjson + "\n" + errorScan.ndjson);
for (const [name, range, file] of [
  ["终点对比", "A1:F45", "endpoints.png"],
  ["NO_MSCA进度", "A1:D29", "progress.png"],
  ["来源与权重", "A1:C15", "sources.png"],
  ["来源与权重", "A17:C50", "bindings.png"],
]) {
  const preview = await workbook.render({ sheetName: name, range, scale: 1.25, format: "png" });
  await fs.writeFile(path.join(previewDir, file), new Uint8Array(await preview.arrayBuffer()));
}
const filename = path.join(outputDir, "实验数据汇总.xlsx");
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(filename);
await fs.writeFile(path.join(buildDir, "workbook_checks.json"), JSON.stringify({
  status: "EXPORTED_AND_FORMULAS_VERIFIED", formula_checks: formulaChecks, input_change_check: true,
  NO_MSCA_final_cells_blank: true, sheets: ["终点对比", "NO_MSCA进度", "来源与权重"],
  export: filename, error_scan: errorScan.ndjson,
}, null, 2) + "\n");
console.log(JSON.stringify({ status: "EXPORTED", output: filename, formulaChecks }));
