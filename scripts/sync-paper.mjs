import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const execute = promisify(execFile);
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const source = process.argv[2];
if (!source) throw new Error('Usage: node scripts/sync-paper.mjs /path/to/paper [--check]');
const paper = resolve(source);
const tex = await readFile(join(paper, 'sections/experiments.tex'), 'utf8');
const methodNames = [
  ['no-memory', 'No Generated Memory'], ['fifo', 'Recent Memory (FIFO)'],
  ['clip', 'CLIP-based Retrieval'], ['storymem', 'StoryMem-style Memory'],
  ['director', 'DirectorAgent (Claude)'], ['reel', 'REEL'],
];
const metric = (key, label, short, description) => ({ key, label, short, description });
const clean = value => value.replace(/\\(?:textbf|underline)\{([^{}]+)\}/g, '$1').replace(/\\\$/g, '$').replace(/\\\s*$/g, '').replace(/~/g, ' ').trim();
function table(label) {
  const position = tex.indexOf(`\\label{${label}}`);
  if (position < 0) throw new Error(`Missing table ${label}`);
  return tex.slice(position, tex.indexOf('\\end{table}', position));
}
function numericRows(content, columns) {
  return content.split('\n').filter(line => line.includes('&')).flatMap(line => {
    const parts = line.trim().replace(/\\\\\s*$/, '').split('&').map(clean);
    if (parts.length !== columns + 1 || !parts.slice(1).every(value => /^(?:\d+(?:\.\d+)?|N\/A)$/.test(value))) return [];
    return [{ name: parts[0], values: parts.slice(1).map(value => value === 'N/A' ? null : Number(value)) }];
  });
}
function benchmark(label, metrics, precision, name, maximum, defaultMetric) {
  const rows = numericRows(table(label), metrics.length);
  if (rows.length !== 12) throw new Error(`${label}: expected 12 rows, found ${rows.length}`);
  const backends = {};
  for (const [backendIndex, backend] of ['echo', 'h3'].entries()) {
    backends[backend] = rows.slice(backendIndex * 6, backendIndex * 6 + 6).map((row, rowIndex) => {
      if (row.name !== methodNames[rowIndex][1]) throw new Error(`Unexpected method: ${row.name}`);
      return { id: methodNames[rowIndex][0], name: row.name, values: Object.fromEntries(metrics.map((item, metricIndex) => [item.key, row.values[metricIndex]])) };
    });
  }
  return { name, precision, maximum, defaultMetric, metrics, backends };
}
const reelMetrics = [
  metric('core', 'Core-100', 'Core', 'Overall visual quality on REEL-Bench.'),
  metric('pf', 'Prompt following', 'PF', 'Shot-level adherence to the requested content.'),
  metric('ref', 'Reference alignment', 'Ref.', 'Alignment with available references, when applicable.'),
  metric('vq', 'Visual quality', 'VQ', 'Shot-level visual quality.'),
  metric('motion', 'Motion quality', 'Motion', 'Shot-level motion quality.'),
  metric('cons', 'Cross-shot consistency', 'Cons.', 'Story-level cross-shot visual consistency.'),
  metric('state', 'State compliance', 'State', 'Adherence to requested state changes, when applicable.'),
  metric('narr', 'Narrative coherence', 'Narr.', 'Story-level narrative coherence.'),
];
const stMetrics = [
  metric('aesthetic', 'Aesthetic quality', 'Aesthetic', 'LAION aesthetic quality on ST-Bench.'),
  metric('globalPf', 'Global prompt following', 'Global PF', 'ViCLIP-based global prompt following.'),
  metric('shotPf', 'Shot prompt following', 'Shot PF', 'ViCLIP-based shot-level prompt following.'),
  metric('cons', 'Cross-shot consistency', 'Cons.', 'Overall cross-shot consistency on ST-Bench.'),
  metric('top10', 'Top-10 consistency', 'Top-10', 'Consistency of the top-10 text-selected shot pairs.'),
];
const training = numericRows(table('tab:reel_ablations').split('\\end{subtable}')[0], 3);
const memory = numericRows(table('tab:memory_ablations').split('\\end{subtable}')[0], 2);
const transfer = numericRows(table('tab:backend_transfer_ablation').split('\\end{subtable}')[0], 2);
for (const [name, rows] of Object.entries({ training, memory, transfer })) {
  if (rows.length !== 4) throw new Error(`${name}: expected four rows, found ${rows.length}`);
}
const results = {
  source: { document: 'REEL preprint', tables: ['1', '2', '3(a)', '3(b)', '3(c)'], sha256: createHash('sha256').update(tex).digest('hex'), scope: 'Values reported in the accompanying manuscript. Quality scores include only complete, decodable stories; average completed shots includes all attempts.' },
  backendNames: { echo: 'JoyAI-Echo 1.5', h3: 'MiniMax-H3' },
  benchmarks: {
    reel: benchmark('tab:reelbench_results', reelMetrics, 2, 'REEL-Bench', 100, 'core'),
    st: benchmark('tab:stbench_results', stMetrics, 3, 'ST-Bench', 1, 'cons'),
  },
  training: training.map(row => ({ name: row.name, shot: row.values[0], story: row.values[1], completedShots: row.values[2] })),
  memory: memory.map(row => ({ name: row.name, shot: row.values[0], story: row.values[1] })),
  transfer: transfer.map(row => ({ name: row.name, shot: row.values[0], story: row.values[1] })),
};
const resultsPath = join(root, 'docs/results.json');
const serialized = JSON.stringify(results, null, 2) + '\n';
if (process.argv.includes('--check')) {
  if (await readFile(resultsPath, 'utf8') !== serialized) throw new Error('Website scores differ from the manuscript.');
  const cells = [...Object.values(results.benchmarks).flatMap(benchmark => Object.values(benchmark.backends).flatMap(rows => rows.flatMap(row => Object.values(row.values)))), ...results.training.flatMap(row => [row.shot, row.story, row.completedShots]), ...results.memory.flatMap(row => [row.shot, row.story]), ...results.transfer.flatMap(row => [row.shot, row.story])];
  console.log(`Verified ${cells.filter(value => value !== null).length} numerical cells and ${cells.filter(value => value === null).length} N/A cells against manuscript Tables 1–3.`);
} else {
  await mkdir(join(root, 'docs/assets'), { recursive: true });
  await writeFile(resultsPath, serialized);
  await copyFile(join(paper, 'main.pdf'), join(root, 'docs/assets/reel.pdf'));
  const figures = [['teaser', 'teaser'], ['framework', 'framework'], ['multishot', 'multishot'], ['data_benchmark', 'data-benchmark']];
  await Promise.all(figures.map(async ([input, output]) => {
    await execute('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', join(paper, `output/figures/${input}.png`), '-vf', 'scale=min(1800\\,iw):-2', '-c:v', 'libwebp', '-quality', '90', '-compression_level', '6', '-threads', '2', join(root, `docs/assets/${output}.webp`)]);
    console.log(`Published web asset: ${output}.webp`);
  }));
  const cases = [['high-pine', 108, 228], ['courier', 376, 228], ['guardian', 644, 228], ['ribbon', 912, 224]];
  await Promise.all(cases.map(async ([name, top, height]) => {
    await execute('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', join(paper, 'output/figures/teaser.png'), '-vf', `crop=iw*1552/1600:ih*${height}/1136:iw*24/1600:ih*${top}/1136,scale=1800:-2`, '-c:v', 'libwebp', '-quality', '92', '-threads', '2', join(root, `docs/assets/case-${name}.webp`)]);
    console.log(`Published paper case: ${name}`);
  }));
  console.log('Synchronized manuscript table values, paper PDF, figures, and four paper-case crops.');
}
