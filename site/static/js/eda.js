/* EDA: графики «число объектов по времени» из #eda-data. Цвет закреплён за типом объекта. */
(function () {
  var el = document.getElementById('eda-data');
  if (!el || !window.TE || !TE.LineChart) return;
  var videos = [];
  try { videos = JSON.parse(el.textContent || '[]'); } catch (err) { return; }
  var PALETTE = { car: '#3987e5', person: '#d95926', truck: '#c98500', bus: '#9085e9', motorcycle: '#d55181', bicycle: '#199e70', vehicle: '#3987e5', pedestrian: '#d95926' };
  var FALLBACK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
  document.querySelectorAll('[data-eda-chart]').forEach(function (canvas) {
    var v = videos[+canvas.getAttribute('data-eda-chart')];
    var cot = v && v.counts_over_time;
    if (!cot || !cot.t) return;
    var k = 0;
    var series = Object.keys(cot).filter(function (key) { return key !== 't'; }).map(function (key) {
      var color = PALETTE[key] || FALLBACK[k++ % FALLBACK.length];
      return { name: key, values: cot[key].map(Number), color: color };
    });
    new TE.LineChart(canvas, { x: cot.t.map(Number), series: series, yMin: 0, xMin: 0 });
  });
})();
