/* Небольшой линейный график на canvas: сетка, оси, несколько серий, легенда,
   наведение (перекрестие + подсказка), опционально порог и «плейхед» для синхронизации с видео. */
(function () {
  var TE = window.TE = window.TE || {};
  var INK = '#e6e9ef', INK2 = '#8b94a7', GRID = '#232b3a';

  function LineChart(canvas, opts) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.o = Object.assign({
      x: [], series: [], yMin: 0, yMax: null, xMin: 0, xMax: null,
      legend: true, threshold: null, thresholdLabel: 'threshold', playhead: null, onSeek: null, fill: false,
      pad: { l: 42, r: 12, t: 12, b: 26 },
      xFormat: function (v) { return TE.fmtTime(v); },
      yFormat: function (v) { return (Math.abs(v) >= 100 || v === Math.round(v)) ? String(Math.round(v)) : v.toFixed(2); }
    }, opts || {});
    this.hoverX = null;
    this.tip = document.createElement('div');
    this.tip.className = 'chart-tip';
    if (canvas.parentNode) canvas.parentNode.appendChild(this.tip);
    this._bind();
    this.resize();
    var self = this;
    if (window.ResizeObserver) new ResizeObserver(function () { self.resize(); }).observe(canvas);
    else window.addEventListener('resize', function () { self.resize(); });
  }

  LineChart.prototype.setData = function (x, series) { this.o.x = x; this.o.series = series; this.draw(); };
  LineChart.prototype.setPlayhead = function (t) {
    if (this.o.playhead !== null && Math.abs(this.o.playhead - t) < 0.02) return;
    this.o.playhead = t; this.draw();
  };
  LineChart.prototype.setXMax = function (v) { this.o.xMax = v; this.draw(); };

  LineChart.prototype.resize = function () {
    var r = this.canvas.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    this.w = Math.max(60, r.width);
    this.h = Math.max(40, r.height);
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  };

  LineChart.prototype._range = function () {
    var o = this.o, xs = o.x;
    var xmin = o.xMin != null ? o.xMin : (xs.length ? xs[0] : 0);
    var xmax = o.xMax != null ? o.xMax : (xs.length ? xs[xs.length - 1] : 1);
    if (xmax <= xmin) xmax = xmin + 1;
    var ymin = o.yMin != null ? o.yMin : Infinity, ymax = o.yMax;
    if (ymax == null || o.yMin == null) {
      var lo = Infinity, hi = -Infinity;
      o.series.forEach(function (s) { s.values.forEach(function (v) { if (v != null && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }); });
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      if (o.yMin == null) ymin = Math.min(0, lo);
      if (ymax == null) { ymax = hi <= ymin ? ymin + 1 : hi; var step = TE.niceStep(ymax - ymin, 4); ymax = Math.ceil(ymax / step) * step; }
    }
    this.xmin = xmin; this.xmax = xmax; this.ymin = ymin; this.ymax = ymax;
  };

  LineChart.prototype.x2px = function (x) { var p = this.o.pad; return p.l + (x - this.xmin) / (this.xmax - this.xmin) * (this.w - p.l - p.r); };
  LineChart.prototype.y2px = function (y) { var p = this.o.pad; return this.h - p.b - (y - this.ymin) / (this.ymax - this.ymin) * (this.h - p.t - p.b); };
  LineChart.prototype.px2x = function (px) {
    var p = this.o.pad, pw = this.w - p.l - p.r;
    if (px < p.l - 4 || px > this.w - p.r + 4) return null;
    return Math.max(this.xmin, Math.min(this.xmax, this.xmin + (px - p.l) / pw * (this.xmax - this.xmin)));
  };

  LineChart.prototype._nearest = function (x) {
    var xs = this.o.x, lo = 0, hi = xs.length - 1;
    if (!xs.length) return -1;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (xs[mid] < x) lo = mid + 1; else hi = mid; }
    if (lo > 0 && Math.abs(xs[lo - 1] - x) < Math.abs(xs[lo] - x)) lo--;
    return lo;
  };

  LineChart.prototype._bind = function () {
    var self = this, c = this.canvas;
    function pos(e) { var r = c.getBoundingClientRect(); var t = e.touches ? e.touches[0] : e; return { x: t.clientX - r.left, y: t.clientY - r.top }; }
    c.addEventListener('mousemove', function (e) { self.hoverX = pos(e).x; self.draw(); });
    c.addEventListener('mouseleave', function () { self.hoverX = null; self.tip.style.display = 'none'; self.draw(); });
    c.addEventListener('click', function (e) { if (self.o.onSeek) { var t = self.px2x(pos(e).x); if (t != null) self.o.onSeek(t); } });
    c.addEventListener('touchstart', function (e) { self.hoverX = pos(e).x; self.draw(); }, { passive: true });
  };

  LineChart.prototype.draw = function () {
    var ctx = this.ctx, o = this.o, p = o.pad, w = this.w, h = this.h;
    this._range();
    ctx.clearRect(0, 0, w, h);
    ctx.font = '11px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    // сетка Y
    var ystep = TE.niceStep(this.ymax - this.ymin, 4);
    ctx.strokeStyle = GRID; ctx.lineWidth = 1; ctx.fillStyle = INK2; ctx.textAlign = 'right';
    for (var y = Math.ceil(this.ymin / ystep) * ystep; y <= this.ymax + 1e-9; y += ystep) {
      var py = Math.round(this.y2px(y)) + 0.5;
      ctx.beginPath(); ctx.moveTo(p.l, py); ctx.lineTo(w - p.r, py); ctx.stroke();
      ctx.fillText(o.yFormat(y), p.l - 6, py);
    }
    // сетка X
    var xstep = TE.niceStep(this.xmax - this.xmin, Math.max(2, Math.floor((w - p.l - p.r) / 90)));
    ctx.textAlign = 'center';
    for (var x = Math.ceil(this.xmin / xstep) * xstep; x <= this.xmax + 1e-9; x += xstep) {
      var px = Math.round(this.x2px(x)) + 0.5;
      ctx.beginPath(); ctx.moveTo(px, p.t); ctx.lineTo(px, h - p.b); ctx.stroke();
      ctx.fillText(o.xFormat(x), px, h - p.b / 2);
    }
    // порог
    if (o.threshold != null) {
      var ty = Math.round(this.y2px(o.threshold)) + 0.5;
      ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = INK2;
      ctx.beginPath(); ctx.moveTo(p.l, ty); ctx.lineTo(w - p.r, ty); ctx.stroke(); ctx.restore();
      ctx.textAlign = 'left'; ctx.fillStyle = INK2; ctx.fillText(o.thresholdLabel + ' ' + o.threshold, p.l + 4, ty - 8);
    }
    // серии
    var self = this;
    ctx.save(); ctx.beginPath(); ctx.rect(p.l, p.t, w - p.l - p.r, h - p.t - p.b); ctx.clip();
    o.series.forEach(function (s) {
      var xs = o.x, vs = s.values;
      ctx.beginPath();
      var started = false;
      for (var i = 0; i < xs.length; i++) {
        var v = vs[i]; if (v == null || !isFinite(v)) { started = false; continue; }
        var X = self.x2px(xs[i]), Y = self.y2px(v);
        if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
      }
      ctx.strokeStyle = s.color || TE.color('illegal_u_turn'); ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();
      if (o.fill && xs.length) {
        ctx.lineTo(self.x2px(xs[xs.length - 1]), self.y2px(self.ymin)); ctx.lineTo(self.x2px(xs[0]), self.y2px(self.ymin)); ctx.closePath();
        ctx.globalAlpha = 0.12; ctx.fillStyle = s.color; ctx.fill(); ctx.globalAlpha = 1;
      }
    });
    ctx.restore();
    // плейхед
    if (o.playhead != null && o.playhead >= this.xmin && o.playhead <= this.xmax) {
      var ph = Math.round(this.x2px(o.playhead)) + 0.5;
      ctx.strokeStyle = '#5b9cff'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(ph, p.t); ctx.lineTo(ph, h - p.b); ctx.stroke();
    }
    // легенда
    if (o.legend && o.series.length > 1) {
      ctx.textAlign = 'left'; var lx = p.l + 8, ly = p.t + 8;
      o.series.forEach(function (s) {
        ctx.fillStyle = s.color; ctx.fillRect(lx, ly - 4, 10, 8);
        ctx.fillStyle = INK; ctx.fillText(s.name, lx + 14, ly);
        lx += 14 + ctx.measureText(s.name).width + 14;
      });
    }
    // наведение
    this.tip.style.display = 'none';
    if (this.hoverX != null && o.x.length) {
      var hx = this.px2x(this.hoverX);
      if (hx != null) {
        var i = this._nearest(hx), cx = Math.round(this.x2px(o.x[i])) + 0.5;
        ctx.strokeStyle = INK2; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(cx, p.t); ctx.lineTo(cx, h - p.b); ctx.stroke(); ctx.setLineDash([]);
        var lines = ['<b>' + o.xFormat(o.x[i]) + '</b>'];
        o.series.forEach(function (s) {
          var v = s.values[i]; if (v == null) return;
          ctx.beginPath(); ctx.arc(cx, self.y2px(v), 3.5, 0, Math.PI * 2); ctx.fillStyle = s.color; ctx.fill();
          lines.push('<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:' + s.color + ';margin-right:6px"></span>' + (s.name || '') + ' ' + o.yFormat(v));
        });
        this.tip.innerHTML = lines.join('<br>');
        this.tip.style.display = 'block';
        var box = this.canvas.parentNode.getBoundingClientRect(), cr = this.canvas.getBoundingClientRect();
        var left = cr.left - box.left + cx + 12;
        if (left + this.tip.offsetWidth > box.width - 4) left = cr.left - box.left + cx - this.tip.offsetWidth - 12;
        this.tip.style.left = Math.max(0, left) + 'px';
        this.tip.style.top = (cr.top - box.top + p.t) + 'px';
      }
    }
  };

  TE.LineChart = LineChart;
})();
