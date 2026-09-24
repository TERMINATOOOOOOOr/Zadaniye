/* Общий словарь классов: цвета и подписи приходят из config.py через window.TE_CLASSES. */
(function () {
  var TE = window.TE = window.TE || {};
  var C = window.TE_CLASSES || {};
  TE.classes = C;
  TE.classOrder = Object.keys(C);
  TE.color = function (id) { return (C[id] && C[id].color) || '#8b94a7'; };
  TE.desc = function (id) { return (C[id] && C[id].desc) || id; };
  TE.badge = function (id) {
    var s = document.createElement('span');
    s.className = 'badge';
    s.style.setProperty('--c', TE.color(id));
    s.textContent = id;
    s.title = TE.desc(id);
    return s;
  };
  TE.fmtTime = function (sec) {
    sec = Math.max(0, +sec || 0);
    var m = Math.floor(sec / 60), s = sec - m * 60;
    return m + ':' + (s < 10 ? '0' : '') + s.toFixed(1);
  };
  TE.el = function (tag, attrs, children) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'class') e.className = attrs[k];
      else if (k === 'text') e.textContent = attrs[k];
      else e.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) { e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return e;
  };
  TE.niceStep = function (range, target) {
    var raw = range / Math.max(1, target || 6);
    if (!(raw > 0)) return 1;
    var pow = Math.pow(10, Math.floor(Math.log10(raw)));
    var f = raw / pow;
    var m = f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10;
    return m * pow;
  };
})();
