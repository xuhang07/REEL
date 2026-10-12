const select = selector => document.querySelector(selector);
const selectAll = selector => [...document.querySelectorAll(selector)];
const state = { benchmark: 'reel', backend: 'echo', metric: 'core', comparison: 'no-memory' };
let results;
const format = (value, precision = 2) => value === null ? 'N/A' : value.toFixed(precision);
const escapeHtml = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);

function setupTabs(selector, callback) {
  const buttons = selectAll(selector);
  for (const button of buttons) {
    button.addEventListener('click', () => {
      for (const sibling of buttons) {
        sibling.setAttribute('aria-selected', String(sibling === button));
        sibling.tabIndex = sibling === button ? 0 : -1;
      }
      callback(button);
    });
    button.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const current = buttons.indexOf(button);
      const target = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (current + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
      buttons[target].focus();
      buttons[target].click();
    });
  }
}

const figures = {
  framework: { image: 'assets/framework.webp', alt: 'Single-shot inference with the REEL memory planner and a backend-specific adapter.', caption: 'REEL uses tools to retrieve and create evidence, submit an evidence plan, review a generated candidate, and optionally update persistent memory.' },
  multishot: { image: 'assets/multishot.webp', alt: 'Multi-shot inference sharing memory-bank, registry, and previous-state resources between shots.', caption: 'Each shot starts a new interaction while persistent resources carry information forward. In this paper example, Shot 5 stores a neighbor reference for retrieval and reuse in Shot 8; Shot 7 is omitted.' },
};
setupTabs('[data-figure]', button => {
  const figure = figures[button.dataset.figure];
  const image = select('#method-image');
  image.src = figure.image;
  image.alt = figure.alt;
  const zoom = image.closest('[data-image]');
  zoom.dataset.image = figure.image;
  zoom.dataset.caption = figure.caption;
  select('#method-caption').textContent = figure.caption;
  select('#method-figure').setAttribute('aria-labelledby', button.id);
});

const dialog = select('#figure-dialog');
for (const button of selectAll('[data-image]')) {
  button.addEventListener('click', () => {
    select('#dialog-image').src = button.dataset.image;
    select('#dialog-image').alt = button.dataset.caption;
    select('#dialog-caption').textContent = button.dataset.caption;
    dialog.showModal();
  });
}
select('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

select('#copy-bibtex').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(select('#bibtex').textContent);
    select('#copy-status').textContent = 'BibTeX copied to clipboard.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(select('#bibtex'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    select('#copy-status').textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});

function renderHeadlines() {
  const cards = [];
  for (const backend of ['echo', 'h3']) {
    const rows = results.benchmarks.reel.backends[backend];
    const gain = rows.find(row => row.id === 'reel').values.core - rows.find(row => row.id === 'no-memory').values.core;
    cards.push({ backend, score: `+${gain.toFixed(2)}`, unit: 'pts', name: 'REEL-Bench · Core-100', note: 'vs. No Generated Memory' });
  }
  for (const backend of ['echo', 'h3']) {
    const rows = results.benchmarks.st.backends[backend];
    const baseline = rows.find(row => row.id === 'no-memory').values.cons;
    const gain = (rows.find(row => row.id === 'reel').values.cons / baseline - 1) * 100;
    cards.push({ backend, score: `+${gain.toFixed(1)}%`, unit: '', name: 'ST-Bench · Consistency', note: 'relative gain vs. No Generated Memory' });
  }
  select('#headline-results').innerHTML = cards.map(card => `<article class="headline-result"><span>${results.backendNames[card.backend]}</span><strong>${card.score} <em>${card.unit}</em></strong><span>${card.name}</span><small>${card.note}</small></article>`).join('');
}


const displayedMethods = ['no-memory', 'director', 'reel'];
const displayedMetrics = { reel: ['core', 'pf', 'state', 'narr'], st: ['cons', 'top10'] };
for (const button of selectAll('[data-case-filter]')) {
  button.addEventListener('click', () => {
    for (const control of selectAll('[data-case-filter]')) control.setAttribute('aria-pressed', String(control === button));
    for (const card of selectAll('[data-case-backend]')) card.hidden = button.dataset.caseFilter !== 'all' && card.dataset.caseBackend !== button.dataset.caseFilter;
  });
}

function scoreRange(values, maximum) {
  const minimumScore = Math.min(...values);
  const maximumScore = Math.max(...values);
  const spread = Math.max(maximumScore - minimumScore, maximum / 100);
  const step = maximum === 100 ? (spread > 6 ? 2 : 1) : (spread > .05 ? .02 : .01);
  const minimum = Math.max(0, Math.floor((minimumScore - spread * .08) / step) * step);
  const upper = Math.min(maximum, Math.ceil((maximumScore + spread * .08) / step) * step);
  return { minimum, maximum: upper };
}

function renderMainChart() {
  const benchmark = results.benchmarks[state.benchmark];
  const metric = benchmark.metrics.find(item => item.key === state.metric);
  const rows = benchmark.backends[state.backend].filter(row => displayedMethods.includes(row.id));
  const range = scoreRange(rows.map(row => row.values[state.metric]), benchmark.maximum);
  const width = range.maximum - range.minimum;
  const precision = benchmark.maximum === 100 ? 0 : 3;
  select('#chart-title').textContent = metric.label;
  select('#chart-description').textContent = metric.description;
  select('#zoom-label').textContent = `Zoomed score axis · ${format(range.minimum, precision)}–${format(range.maximum, precision)}`;
  select('#main-chart').dataset.minimum = String(range.minimum);
  select('#main-chart').dataset.maximum = String(range.maximum);
  select('#main-chart').setAttribute('aria-label', `${benchmark.name}, ${results.backendNames[state.backend]}, ${metric.label}. Zoomed score axis from ${range.minimum} to ${range.maximum}; higher is better.`);
  select('#main-chart').innerHTML = rows.map(row => {
    const value = row.values[state.metric];
    const position = (value - range.minimum) / width * 100;
    const name = row.id === 'director' ? 'DirectorAgent' : row.name;
    return `<div class="dot-row ${row.id === 'reel' ? 'reel-dot' : row.id === 'director' ? 'director-dot' : ''}"><span class="dot-label">${escapeHtml(name)}</span><span class="dot-track" aria-hidden="true"><span class="score-dot" style="--dot-position:${position}%"></span></span><span class="dot-value">${format(value, benchmark.precision)}</span></div>`;
  }).join('');
  select('#chart-axis').innerHTML = [0, .25, .5, .75, 1].map(ratio => `<span>${format(range.minimum + ratio * width, precision)}</span>`).join('');
  const reel = rows.find(row => row.id === 'reel').values[state.metric];
  select('#score-gains').innerHTML = rows.filter(row => row.id !== 'reel').map((row, index) => {
    const difference = reel - row.values[state.metric];
    const display = state.benchmark === 'st' ? `${((reel / row.values[state.metric] - 1) * 100).toFixed(1)}%` : format(difference, 2);
    const label = row.id === 'no-memory' ? 'REEL vs. No Memory' : 'REEL vs. DirectorAgent';
    return `<div class="score-gain ${index ? 'secondary' : ''}"><span>${label}</span><strong>${difference >= 0 ? '+' : ''}${display} <small>${state.benchmark === 'st' ? 'relative' : 'pts'}</small></strong></div>`;
  }).join('');
}

function updateMetricOptions() {
  const benchmark = results.benchmarks[state.benchmark];
  select('#metric-select').innerHTML = benchmark.metrics.filter(metric => displayedMetrics[state.benchmark].includes(metric.key)).map(metric => `<option value="${metric.key}">${escapeHtml(metric.label)}</option>`).join('');
  select('#metric-select').value = state.metric;
}

try {
  const response = await fetch(new URL('results.json', import.meta.url));
  if (!response.ok) throw new Error(`Could not load results (${response.status}).`);
  results = await response.json();
  renderHeadlines();
  updateMetricOptions();
  renderMainChart();
  setupTabs('[data-benchmark]', button => {
    state.benchmark = button.dataset.benchmark;
    state.metric = results.benchmarks[state.benchmark].defaultMetric;
    select('#benchmark-panel').setAttribute('aria-labelledby', button.id);
    updateMetricOptions();
    renderMainChart();
  });
  select('#backend-select').addEventListener('change', event => { state.backend = event.target.value; renderMainChart(); });
  select('#metric-select').addEventListener('change', event => { state.metric = event.target.value; renderMainChart(); });
} catch (error) {
  select('#headline-results').innerHTML = '<p class="loading-placeholder">Please see the paper PDF for the full results.</p>';
  select('#results-status').textContent = error.message;
}
