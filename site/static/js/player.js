/* Связка «видео + таймлайн + таблица событий + кривая риска» для страниц результатов и демо.
   root должен содержать элементы с data-role=video|timeline|risk|table. */
(function () {
  var TE = window.TE = window.TE || {};

  TE.mountResult = function (root, data) {
    var video = root.querySelector('[data-role=video]');
    var tlEl = root.querySelector('[data-role=timeline]');
    var riskEl = root.querySelector('[data-role=risk]');
    var tableEl = root.querySelector('[data-role=table]');
    var events = (data.events || []).map(function (e) { return { s: +e[0], e: +e[1], label: String(e[2]) }; })
      .sort(function (a, b) { return a.s - b.s; });
    var risk = data.risk || [];
    var duration = +data.duration || 0;
    if (!duration && events.length) duration = Math.max.apply(null, events.map(function (e) { return e.e; }));
    if (!duration && risk.length) duration = +risk[risk.length - 1][0];

    function seek(t) {
      if (!video) return;
      var max = (isFinite(video.duration) && video.duration) ? video.duration : duration;
      try {
        video.currentTime = Math.max(0, Math.min(t, Math.max(0, max - 0.05)));
        var p = video.play(); if (p && p.catch) p.catch(function () {});
      } catch (err) { /* видео ещё не готово */ }
      update();
    }

    var tl = tlEl ? new TE.Timeline(tlEl, { events: events, duration: duration, onSeek: seek }) : null;
    var chart = null;
    if (riskEl) {
      if (risk.length) {
        chart = new TE.LineChart(riskEl, {
          x: risk.map(function (p) { return +p[0]; }),
          series: [{ name: 'risk', values: risk.map(function (p) { return +p[1]; }), color: TE.color('accident') }],
          yMin: 0, yMax: 1, xMin: 0, xMax: duration, threshold: 0.5, fill: true, legend: false, onSeek: seek,
          yFormat: function (v) { return v.toFixed(1); }
        });
      } else {
        var box = riskEl.parentNode;
        var note = TE.el('p', { class: 'muted small', text: 'no risk curve available' });
        note.style.margin = '6px 8px';
        box.replaceChild(note, riskEl);
      }
    }

    if (tableEl) {
      tableEl.innerHTML = '';
      if (!events.length) {
        tableEl.appendChild(TE.el('tbody', null, [TE.el('tr', null, [TE.el('td', { class: 'muted', text: 'no events found' })])]));
      } else {
        tableEl.appendChild(TE.el('thead', null, [TE.el('tr', null, ['#', 'class', 'start', 'end', 'length'].map(function (h) { return TE.el('th', { text: h }); }))]));
        var tb = TE.el('tbody');
        events.forEach(function (e, i) {
          var tr = TE.el('tr', { 'data-t': e.s, title: 'jump to ' + TE.fmtTime(e.s) }, [
            TE.el('td', { class: 'num', text: String(i + 1) }),
            TE.el('td', null, [TE.badge(e.label)]),
            TE.el('td', { class: 'num', text: TE.fmtTime(e.s) }),
            TE.el('td', { class: 'num', text: TE.fmtTime(e.e) }),
            TE.el('td', { class: 'num', text: (e.e - e.s).toFixed(1) + ' s' })
          ]);
          tr.addEventListener('click', function () { seek(e.s); });
          tb.appendChild(tr);
        });
        tableEl.appendChild(tb);
      }
    }

    var raf = null;
    function update() {
      if (!video) return;
      var t = video.currentTime || 0;
      if (tl) tl.setTime(t);
      if (chart) chart.setPlayhead(t);
    }
    function loop() { update(); if (video && !video.paused && !video.ended) raf = requestAnimationFrame(loop); }
    if (video) {
      video.addEventListener('loadedmetadata', function () {
        if (!duration && isFinite(video.duration) && video.duration) {
          duration = video.duration;
          if (tl) tl.setDuration(duration);
          if (chart) chart.setXMax(duration);
        }
      });
      ['timeupdate', 'seeked', 'pause'].forEach(function (ev) { video.addEventListener(ev, update); });
      video.addEventListener('play', function () { cancelAnimationFrame(raf); loop(); });
    }
    return { seek: seek, timeline: tl, chart: chart };
  };
})();
