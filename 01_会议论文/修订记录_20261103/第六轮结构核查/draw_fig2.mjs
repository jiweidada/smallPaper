import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const sharp = require('C:/Users/jiwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const here = path.dirname(fileURLToPath(import.meta.url));
const source = path.join(here, '..', 'figures', 'fig2_mps.svg');
const svgOut = path.join(here, 'fig2_mps_sixth_candidate.svg');
const pngOut = path.join(here, 'fig2_mps_sixth_candidate.png');
if (fs.existsSync(svgOut) || fs.existsSync(pngOut)) throw new Error('Candidate already exists; inspect it first.');
const svg = fs.readFileSync(source, 'utf8');
const before = '<text x="20.0" y="414.99999999999994" dominant-baseline="central" text-anchor="start" font-family="Arial" font-size="14.57222222222222">Conv blocks include BN and the configured activation. DWConv denotes depthwise convolution.</text>';
const after = '<text x="20" y="398" dominant-baseline="central" text-anchor="start" font-family="Arial" font-size="14.57">ConvBNAct: convolution → BN → activation → learned scalar affine (use_lab=True).</text>' +
              '<text x="20" y="417" dominant-baseline="central" text-anchor="start" font-family="Arial" font-size="14.57">Detail, main, fusion and final use ConvBNAct; high-pass and gate omit the affine. DWConv: depthwise.</text>';
if (svg.split(before).length !== 2) throw new Error('Expected exactly one original legend.');
const candidate = svg.replace(before, after);
fs.writeFileSync(svgOut, candidate, {flag: 'wx'});
const info = await sharp(Buffer.from(candidate), {density: 172.8}).png().toFile(pngOut);
if (info.width !== 2400 || info.height !== 1032) throw new Error(`Unexpected dimensions ${info.width}x${info.height}`);
console.log(JSON.stringify({svgOut, pngOut, width: info.width, height: info.height}));
