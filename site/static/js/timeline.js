/* Интерактивный таймлайн событий на canvas: дорожка на класс, полосы событий,
   плейхед, наведение с подсказкой, клик — переход к моменту (onSeek). */
(function () {
  var TE = window.TE = window.TE || {};

  function Timeline(container, opts) {
    this.o = Object.assign({ events: [], duration: 0, onSeek: null, laneH: 24, axisH: 24, gutter: 130 }, opts || {});
    this.container = container;
    container.classList.add('timeline');
    this.canvas = document.createElement('canvas');
    this.tip = document.createElement('div');
    this.tip.className = 'tl-tip';
    container.appendChild(this.canvas);
    container.appendChild(this.tip);
    this.ctx = this.canvas.getContext('2d');
    this.t = 0;
    this.hover = null;
    this._lanes();
    this._bind();
    this.resize();
    var self = this;
    if (window.ResizeObserver) new ResizeObserver(function () { self.resize(); }).observe(container);
    else window.addEventListener('resize', function () { self.resize(); });
  }

  Timeline.prototype._lanes = function () {
    var present = {};
    this.o.events.forEach(function (e) { present[e.label] = true; });
    var order = TE.classOrder.filter(function (c) { return present[c]; });
    Object.keys(present).forEach(function (c) { if (order.indexOf(c) < 0) order.push(c); });
    this.lanes = order.length ? order : [];
  };

  Timeline.prototype.setEvents = function (events) { this.o.events = events; this._lanes(); this.resize(); };
  Timeline.prototype.setDuration = function (d) { this.o.duration = d; this.draw(); };
  Timeline.prototype.setTime = function (t) { if (Math.abs(t - this.t) < 0.02) return; this.t = t; this.draw(); };

  Timeline.prototype.resize = function () {
    var dpr = window.devicePixelRatio || 1;
    var w = this.container.getBoundingClientRect().width || 300;
    this.gutter = Math.min(this.o.gutter, Math.max(70, Math.round(w * 0.22)));
    var h = Math.max(1, this.lanes.length) * this.o.laneH + this.o.axisH;
    this.w = w; this.h = h;
    this.container.style.height = h + 'px';
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  };

  Timeline.prototype.x2px = function (t) { return this.gutter + (t / Math.max(1e-6, this.o.duration)) * (this.w - this.gutter - 8); };
  Timeline.prototype.px2t = function (px) { return Math.max(0, Math.min(this.o.duration, (px - this.gutter) / (this.w - this.gutter - 8) * this.o.duration)); };

  Timeline.prototype.hit = function (px, py) {
    if (px < this.gutter) return null;
    var lane = Math.floor(py / this.o.laneH);
    if (lane < 0 || lane >= this.lanes.length) return null;
    var label = this.lanes[lane], t = this.px2t(px), best = null;
    for (var i = 0; i < this.o.events.length; i++) {
      var e = this.o.events[i];
      if (e.label !== label) continue;
      var x1 = this.x2px(e.s), x2 = Math.max(this.x2px(e.e), x1 + 3);
      if (px >= x1 - 2 && px <= x2 + 2) { best = e; break; }
    }
    return best || (t != null ? null : null);
  };

  Timeline.prototype._bind = function () {
    var self = this, c = this.canvas;
    function pos(e) { var r = c.getBoundingClientRect(); var t = e.changedTouches ? e.changedTouches[0] : e; return { x: t.clientX - r.left, y: t.clientY - r.top }; }
    c.addEventListener('mousemove', function (e) {
      var p = pos(e); self.hover = self.hit(p.x, p.y);
      if (self.hover) {
        var ev = self.hover;
        self.tip.innerHTML = '<b>' + ev.label + '</b> · ' + TE.fmtTime(ev.s) + ' – ' + TE.fmtTime(ev.e) + ' (' + (ev.e - ev.s).toFixed(1) + ' s)<br><span style="color:#8b94a7">' + TE.desc(ev.label) + '</span>';
        self.tip.style.display = 'block';
        var left = p.x + 12; if (left + self.tip.offsetWidth > self.w - 4) left = p.x - self.tip.offsetWidth - 12;
        self.tip.style.left = Math.max(0, left) + 'px';
        self.tip.style.top = Math.max(0, p.y - self.tip.offsetHeight - 10) + 'px';
      } else self.tip.style.display = 'none';
      c.style.cursor = p.x >= self.gutter ? 'pointer' : 'default';
      self.draw();
    });
    c.addEventListener('mouseleave', function () { self.hover = null; self.tip.style.display = 'none'; self.draw(); });
    function seekAt(p) {
      if (!self.o.onSeek || p.x < self.gutter) return;
      var ev = self.hit(p.x, p.y);
      self.o.onSeek(ev ? ev.s : self.px2t(p.x));
    }
    c.addEventListener('click', function (e) { seekAt(pos(e)); });
  };

  Timeline.prototype.draw = function () {
    var ctx = this.ctx, w = this.w, h = this.h, laneH = this.o.laneH, self = this;
    ctx.clearRect(0, 0, w, h);
    ctx.font = '11px ui-monospace, Menlo, Consolas, monospace';
    ctx.textBaseline = 'middle';
    // дорожки
    this.lanes.forEach(function (label, i) {
      var y = i * laneH;
      ctx.fillStyle = i % 2 ? 'rgba(255,255,255,0.025)' : 'rgba(255,255,255,0)';
      ctx.fillRect(0, y, w, laneH);
      ctx.fillStyle = TE.color(label); ctx.beginPath(); ctx.arc(10, y + laneH / 2, 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#e6e9ef'; ctx.textAlign = 'left';
      var name = label, maxW = self.gutter - 26;
      while (ctx.measureText(name).width > maxW && name.length > 3) name = name.slice(0, -2) + '…';
      ctx.fillText(name, 20, y + laneH / 2);
    });
    if (!this.lanes.length) {
      ctx.fillStyle = '#8b94a7'; ctx.textAlign = 'left'; ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('no events found', 12, laneH / 2);
      ctx.font = '11px ui-monospace, Menlo, Consolas, monospace';
    }
    // разделитель
    ctx.strokeStyle = '#232b3a'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(this.gutter - 0.5, 0); ctx.lineTo(this.gutter - 0.5, h - this.o.axisH); ctx.stroke();
    // ось времени
    var axisY = h - this.o.axisH, dur = this.o.duration || 1;
    ctx.beginPath(); ctx.moveTo(this.gutter, axisY + 0.5); ctx.lineTo(w - 8, axisY + 0.5); ctx.stroke();
    var step = TE.niceStep(dur, Math.max(2, Math.floor((w - this.gutter) / 80)));
    ctx.fillStyle = '#8b94a7'; ctx.textAlign = 'center';
    for (var t = 0; t <= dur + 1e-9; t += step) {
      var x = Math.round(this.x2px(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, axisY); ctx.lineTo(x, axisY + 4); ctx.stroke();
      ctx.fillText(TE.fmtTime(t), x, axisY + 13);
    }
    // события
    this.o.events.forEach(function (e) {
      var lane = self.lanes.indexOf(e.label); if (lane < 0) return;
      var x1 = self.x2px(e.s), x2 = Math.max(self.x2px(e.e), x1 + 3);
      var y = lane * laneH + 5, bh = laneH - 10;
      ctx.globalAlpha = self.hover && self.hover !== e ? 0.55 : 0.9;
      ctx.fillStyle = TE.color(e.label);
      if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x1, y, x2 - x1, bh, 3); ctx.fill(); } else ctx.fillRect(x1, y, x2 - x1, bh);
      if (self.hover === e) { ctx.globalAlpha = 1; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.strokeRect(x1 + 0.5, y + 0.5, x2 - x1 - 1, bh - 1); }
      ctx.globalAlpha = 1;
    });
    // плейхед
    if (this.t >= 0 && this.t <= dur) {
      var px = Math.round(this.x2px(this.t)) + 0.5;
      ctx.strokeStyle = '#5b9cff'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, axisY); ctx.stroke();
      ctx.fillStyle = '#5b9cff'; ctx.beginPath(); ctx.moveTo(px - 5, axisY); ctx.lineTo(px + 5, axisY); ctx.lineTo(px, axisY + 6); ctx.closePath(); ctx.fill();
    }
  };

  TE.Timeline = Timeline;
})();
