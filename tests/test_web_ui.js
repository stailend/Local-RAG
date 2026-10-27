const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

class Element {
  constructor() {
    this.children = [];
    this.listeners = {};
    this.dataset = {};
    this.value = '';
    this.hidden = false;
    this.files = [];
    const classes = new Set();
    this.classList = {add: name => classes.add(name), remove: name => classes.delete(name),
      contains: name => classes.has(name)};
  }
  set textContent(value) { this.text = value; this.children = []; }
  get textContent() { return this.text; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  setAttribute() {}
  removeAttribute() {}
  contains(target) { return this === target; }
  focus() { this.focused = true; }
  scrollIntoView() { this.scrolled = true; }
}

function setup(fetch) {
  const html = fs.readFileSync(path.join(__dirname, '../src/local_rag/web/index.html'), 'utf8');
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  vm.runInNewContext(script, {
    document: {getElementById: get, createElement: () => new Element(),
      createTextNode: text => text}, fetch, URLSearchParams,
  });
  return get;
}

test('knowledge map selects groups and files for search', async () => {
  const groups = [{id: 'topic-1', label: 'Gardening', files: [
    {source: '/docs/flowers.pdf', summary: 'Flowers and soil'},
  ]}];
  const fetch = async url => ({ok: true, json: async () =>
    url === '/api/clusters' ? {clusters: groups} :
    url === '/api/sources' ? {sources: [], folders: []} :
    {points: 1, collection: 'test', provider: 'local'}});
  const get = setup(fetch);

  await get('explore').listeners.click();
  assert.equal(get('map-panel').hidden, false);
  assert.equal(get('map-panel').scrolled, true);
  assert.equal(get('knowledge-map').children.length, 1);
  const branch = get('knowledge-map').children[0];
  get('source').value = '/docs/other.pdf';
  get('folder').value = '/other';
  branch.children[0].children[0].listeners.click();
  assert.equal(get('active-cluster').dataset.id, 'topic-1');
  assert.equal(get('source').value, '');
  assert.equal(get('folder').value, '');
  assert.equal(get('question').focused, true);
  get('folder').value = '/other';
  branch.children[2].children[0].listeners.click();
  assert.equal(get('source').value, '/docs/flowers.pdf');
  assert.equal(get('folder').value, '');
  assert.equal(get('active-cluster').dataset.id, '');
});

test('drop zone uploads one file and keeps the picker working', async () => {
  const sent = [];
  const fetch = async (url, options) => {
    if (url.startsWith('/api/uploads?')) {
      sent.push({url, file: options.body});
      return {ok: true, json: async () => ({id: 'job-1'})};
    }
    if (url.startsWith('/api/jobs/')) {
      return {ok: true, json: async () => ({stage: 'done', filename: 'file', done: 1, total: 1})};
    }
    return {ok: true, json: async () => url === '/api/sources'
      ? {sources: [], folders: []} : {points: 1, collection: 'test', provider: 'local'}};
  };
  const get = setup(fetch);
  const submit = () => get('upload-form').listeners.submit({preventDefault() {}});
  const drop = files => get('drop-zone').listeners.drop({preventDefault() {}, dataTransfer: {files}});

  drop([{name: 'one.pdf'}, {name: 'two.pdf'}]);
  await submit();
  assert.equal(sent.length, 0);
  assert.equal(get('upload-status').textContent, 'Choose a file first.');

  const dropped = {name: 'diagram.pdf'};
  get('drop-zone').listeners.dragenter({preventDefault() {}});
  assert.equal(get('drop-zone').classList.contains('drag-over'), true);
  drop([dropped]);
  assert.equal(get('drop-zone').classList.contains('drag-over'), false);
  assert.equal(get('file-name').textContent, 'diagram.pdf');
  await submit();
  assert.equal(sent[0].file, dropped);
  assert.match(sent[0].url, /diagram.pdf/);
  assert.equal(get('file-name').textContent, '');

  const picked = {name: 'manual.md'};
  get('file').files = [picked];
  get('file').listeners.change();
  await submit();
  assert.equal(sent[1].file, picked);
});
