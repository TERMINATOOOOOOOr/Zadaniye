/* Страница демо: загрузка файла, опрос задачи каждые 2 с, вывод результата. */
(function () {
  var form = document.getElementById('demo-form');
  if (!form) return;
  var TE = window.TE, L = window.TE_DEMO || { maxMb: 200, maxSec: 120 };
  var input = document.getElementById('demo-file');
  var meta = document.getElementById('file-meta');
  var drop = form.querySelector('.file-drop');
  var submit = document.getElementById('demo-submit');
  var reset = document.getElementById('demo-reset');
  var msg = document.getElementById('demo-msg');
  var status = document.getElementById('job-status');
  var bar = document.getElementById('job-bar');
  var stageEl = document.getElementById('job-stage');
  var pctEl = document.getElementById('job-pct');
  var extraEl = document.getElementById('job-extra');
  var resultEl = document.getElementById('job-result');
  var pollTimer = null;
  var defaultMeta = 'up to ' + L.maxSec + ' s, up to ' + L.maxMb + ' MB';

  function say(text) { msg.textContent = text; msg.hidden = !text; }
  function setBar(pct, stage, extra) {
    status.hidden = false;
    bar.style.width = Math.max(0, Math.min(100, pct)) + '%';
    pctEl.textContent = Math.round(pct) + '%';
    if (stage) stageEl.textContent = stage;
    extraEl.textContent = extra || '';
  }
  function fail(text) {
    say('Error: ' + text);
    bar.style.background = 'var(--bad)';
    submit.disabled = false; reset.hidden = false;
  }
  function fmtMb(b) { return (b / 1048576).toFixed(1) + ' MB'; }

  input.addEventListener('change', function () {
    var f = input.files[0];
    say('');
    if (!f) { meta.textContent = defaultMeta; return; }
    meta.textContent = f.name + ' · ' + fmtMb(f.size);
    probeDuration(f).then(function (d) { if (d) meta.textContent += ' · ' + TE.fmtTime(d); });
  });
  ['dragenter', 'dragover'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add('over'); }); });
  ['dragleave', 'drop'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove('over'); }); });
  drop.addEventListener('drop', function (e) {
    if (e.dataTransfer && e.dataTransfer.files.length) { input.files = e.dataTransfer.files; input.dispatchEvent(new Event('change')); }
  });

  function probeDuration(file) {
    return new Promise(function (res) {
      var v = document.createElement('video');
      v.preload = 'metadata';
      v.onloadedmetadata = function () { var d = v.duration; URL.revokeObjectURL(v.src); res(isFinite(d) ? d : null); };
      v.onerror = function () { res(null); };
      v.src = URL.createObjectURL(file);
    });
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var f = input.files[0];
    say('');
    if (!f) return say('Choose an .mp4 file');
    if (!/\.mp4$/i.test(f.name)) return say('An .mp4 file is required');
    if (f.size > L.maxMb * 1048576) return say('File is larger than ' + L.maxMb + ' MB (' + fmtMb(f.size) + ')');
    submit.disabled = true;
    probeDuration(f).then(function (d) {
      if (d && d > L.maxSec + 0.5) { submit.disabled = false; return say('Video is longer than ' + L.maxSec + ' s (' + TE.fmtTime(d) + ')'); }
      upload(f);
    });
  });

  var urlInput = document.getElementById('demo-url');
  var urlRun = document.getElementById('demo-url-run');
  if (urlRun) urlRun.addEventListener('click', function () {
    var u = (urlInput.value || '').trim();
    say('');
    if (!u) return say('Paste a link to an .mp4 (Google Drive share link or direct URL)');
    if (!/^https?:\/\//i.test(u)) return say('The link must start with http:// or https://');
    urlRun.disabled = true; submit.disabled = true;
    resultEl.hidden = true; bar.style.background = '';
    setBar(0, 'requesting download');
    fetch('/api/jobs_url', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url: u }) })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (x) {
        urlRun.disabled = false;
        if (x.ok && x.j && x.j.id) { setBar(0, 'downloading'); poll(x.j.id); }
        else fail((x.j && x.j.detail) || 'server rejected the link');
      })
      .catch(function () { urlRun.disabled = false; fail('network unavailable'); });
  });

  reset.addEventListener('click', function () {
    clearTimeout(pollTimer);
    form.reset(); input.dispatchEvent(new Event('change')); if (urlInput) urlInput.value = '';
    status.hidden = true; resultEl.hidden = true; reset.hidden = true; submit.disabled = false;
    bar.style.background = ''; bar.style.width = '0%'; say('');
  });

  function upload(f) {
    resultEl.hidden = true;
    bar.style.background = '';
    setBar(0, 'uploading');
    var xhr = new XMLHttpRequest();
    xhr.open('POST', form.getAttribute('action') || '/api/jobs');
    xhr.upload.onprogress = function (e) { if (e.lengthComputable) setBar(e.loaded / e.total * 100, 'uploading'); };
    xhr.onload = function () {
      var j = null; try { j = JSON.parse(xhr.responseText); } catch (err) { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300 && j && j.id) { setBar(0, 'queued'); poll(j.id); }
      else fail((j && j.detail) || ('server responded with ' + xhr.status));
    };
    xhr.onerror = function () { fail('network unavailable'); };
    var fd = new FormData(); fd.append('file', f);
    xhr.send(fd);
  }

  function poll(id) {
    fetch('/api/jobs/' + id, { cache: 'no-store' }).then(function (r) { return r.json(); }).then(function (j) {
      if (j.detail && !j.state) return fail(j.detail);
      var extra = '';
      if (j.state === 'queued' && j.position > 1) extra = '· ' + (j.position - 1) + ' ahead of you';
      if (j.state === 'running' && j.elapsed_sec) extra = '· ' + Math.round(j.elapsed_sec) + ' s';
      setBar(j.progress || 0, j.stage || j.state, extra);
      if (j.state === 'done') showResult(j);
      else if (j.state === 'error') fail(j.error || 'processing failed');
      else pollTimer = setTimeout(function () { poll(id); }, 2000);
    }).catch(function () { pollTimer = setTimeout(function () { poll(id); }, 3000); });
  }

  function showResult(j) {
    var r = j.result || {};
    submit.disabled = false; reset.hidden = false;
    resultEl.hidden = false;
    document.getElementById('res-name').textContent = r.video || j.video || '';
    var timing = r.timing_sec ? ' · Part A ' + r.timing_sec.part_a + ' s, Part B ' + r.timing_sec.part_b + ' s' : '';
    document.getElementById('res-summary').textContent = 'duration ' + TE.fmtTime(r.duration) + ' · events: ' + (r.events || []).length + ' · risk samples: ' + (r.n_risk || 0) + timing;
    var video = resultEl.querySelector('[data-role=video]');
    document.getElementById('res-noannot').hidden = !!r.annotated_url;
    video.src = r.annotated_url || r.input_url;
    var dl = document.getElementById('res-download');
    dl.href = r.events_url || '#';
    var notes = document.getElementById('res-notes');
    notes.innerHTML = '';
    (r.notes || []).forEach(function (n) { notes.appendChild(TE.el('li', { text: n })); });
    // пересоздаём таймлайн/график для нового результата
    var tlEl = resultEl.querySelector('[data-role=timeline]');
    var fresh = TE.el('div', { 'data-role': 'timeline' }); tlEl.parentNode.replaceChild(fresh, tlEl);
    var box = resultEl.querySelector('.chart-box.risk');
    box.innerHTML = '<canvas class="chart" data-role="risk" height="150"></canvas>';
    TE.mountResult(resultEl, r);
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
})();
