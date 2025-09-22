(function(){
  const log = (...a) => { console.log('[APP]', ...a); };

  const req = (path, opts={}) => fetch(path, opts).then(async r => {
    const ct = r.headers.get('content-type')||'';
    const data = ct.includes('application/json') ? await r.json() : await r.text();
    return { ok: r.ok, status: r.status, data };
  });

  const byId = id => {
    const el = document.getElementById(id);
    if(!el) console.error(`[APP] Элемент #${id} не найден в DOM`);
    return el;
  };

  const tabs = document.querySelectorAll('.tab');
  const live = byId('live');
  const add  = byId('add');

  tabs.forEach(t => t.addEventListener('click', () => {
    tabs.forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    const which = t.dataset.tab;
    if(live) live.style.display = which === 'live' ? '' : 'none';
    if(add)  add.style.display  = which === 'add'  ? '' : 'none';
  }));

  const video     = byId('video');
  const overlay   = byId('overlay');
  const ctx       = overlay ? overlay.getContext('2d') : null;
  const startBtn  = byId('startBtn');
  const stopBtn   = byId('stopBtn');
  const statusEl  = byId('status');
  const connEl    = byId('conn');
  const peopleEl  = byId('people');

  let ws = null;
  let run = false;
  let sendTimer = null;

  const setStatus = (t) => { if(statusEl) statusEl.textContent = t; };

  async function listPeople(){
    const r = await req('/api/people');
    log('GET /api/people ->', r.status, r.data);
    if(!r.ok){ setStatus('Ошибка /api/people'); return; }
    peopleEl.innerHTML = '';
    (r.data.people||[]).forEach(p => {
      const span = document.createElement('span');
      span.className='person';
      span.textContent = p;
      peopleEl.appendChild(span);
    });
  }

  function drawBoxes(pred){
    const boxes = pred.boxes || [];
    const names = pred.names || [];
    if(!ctx || !overlay) return;
    ctx.clearRect(0,0,overlay.width, overlay.height);
    ctx.lineWidth = 2; ctx.font = '14px system-ui';
    for(let i=0;i<boxes.length;i++){
      const [l,t,r,b] = boxes[i];
      const name = names[i] || 'Unknown';
      ctx.strokeStyle = name==='Unknown' ? '#ef4444' : '#22c55e';
      ctx.fillStyle = ctx.strokeStyle;
      ctx.strokeRect(l, t, r-l, b-t);
      const label = `${name}`;
      const tw = ctx.measureText(label).width + 8;
      const th = 18;
      ctx.fillRect(l, b-th, tw, th);
      ctx.fillStyle = '#000';
      ctx.fillText(label, l+4, b-4);
    }
  }

  async function start(){
    try{
      if(run) return;
      log('Кнопка Старт нажата');
      run = true; if(startBtn) startBtn.disabled = true; if(stopBtn) stopBtn.disabled = false;
      setStatus('Запуск камеры...');

      // Проверка наличия API getUserMedia
      if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
        setStatus('Браузер не поддерживает getUserMedia');
        log('getUserMedia отсутствует');
        return;
      }

      // Запрос камеры
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 } });
      if(video) video.srcObject = stream;

      await new Promise(res => {
        if(!video){ res(); return; }
        video.onloadedmetadata = () => res();
      });

      if(!overlay || !video){
        setStatus('Видео/канвас не найдены');
        log('video/overlay null');
        return;
      }

      overlay.width  = video.videoWidth  || 640;
      overlay.height = video.videoHeight || 360;

      const wsUrl = (location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/ws';
      log('Открываю WebSocket:', wsUrl);
      ws = new WebSocket(wsUrl);
      ws.onopen = () => {
        if(connEl){ connEl.textContent = 'WS: online'; connEl.style.background = '#0e7490'; }
        log('WebSocket open');
      };
      ws.onclose = (e) => {
        if(connEl){ connEl.textContent = 'WS: offline'; connEl.style.background = '#111827'; }
        log('WebSocket close', e.code, e.reason);
      };
      ws.onerror = (e) => {
        setStatus('Ошибка WebSocket (см. Console)');
        log('WebSocket error', e);
      };
      ws.onmessage = ev => {
        try{
          const msg = JSON.parse(ev.data);
          if(msg.type === 'result') drawBoxes(msg);
        }catch(err){
          log('WS message parse error', err);
        }
      };

      const sendFrame = () => {
        if(!run || !ws || ws.readyState !== 1) return;
        const tmp = document.createElement('canvas');
        tmp.width = overlay.width; tmp.height = overlay.height;
        const tctx = tmp.getContext('2d');
        tctx.drawImage(video, 0, 0, tmp.width, tmp.height);
        const data = tmp.toDataURL('image/jpeg', 0.8);
        ws.send(JSON.stringify({type:'frame', data}));
      };

      sendTimer = setInterval(sendFrame, 100); // ~10 fps
      setStatus('Распознавание запущено. Разрешите камеру, если спросит.');
    }catch(err){
      setStatus('Ошибка при старте (см. Console)');
      log('start() error:', err);
    }
  }

  function stop(){
    log('Кнопка Стоп нажата');
    run = false; if(startBtn) startBtn.disabled = false; if(stopBtn) stopBtn.disabled = true;
    if(sendTimer){ clearInterval(sendTimer); sendTimer = null; }
    if(ws){ try{ ws.close(); }catch(e){} ws = null; }
    if(video && video.srcObject){
      video.srcObject.getTracks().forEach(t => t.stop());
      video.srcObject = null;
    }
    setStatus('Остановлено.');
    if(ctx && overlay) ctx.clearRect(0,0,overlay.width, overlay.height);
  }

  if(startBtn) startBtn.addEventListener('click', start);
  if(stopBtn)  stopBtn.addEventListener('click', stop);

  // ------- Upload (Add person) -------
  const personEl     = byId('person');
  const filesEl      = byId('files');
  const uploadBtn    = byId('uploadBtn');
  const uploadStatus = byId('uploadStatus');

  if(uploadBtn) uploadBtn.addEventListener('click', async () => {
    try{
      log('Кнопка Загрузить нажата');
      const name = (personEl?.value||'').trim();
      if(!name){ uploadStatus.textContent='Укажите имя.'; return; }
      const files = filesEl?.files;
      if(!files || !files.length){ uploadStatus.textContent='Выберите изображения.'; return; }

      const fd = new FormData();
      fd.append('person', name);
      Array.from(files).forEach(f => fd.append('files', f));

      uploadStatus.textContent='Загрузка...';
      const r = await req('/api/known_faces', { method:'POST', body: fd });
      log('POST /api/known_faces ->', r.status, r.data);
      if(r.ok && r.data && r.data.ok){
        uploadStatus.textContent = `ОК: ${r.data.person}, сохранено: ${r.data.saved.length}, энкодингов: ${r.data.encodings}`;
        await listPeople();
      } else {
        uploadStatus.textContent = `Ошибка загрузки (${r.status})`;
      }
    }catch(err){
      uploadStatus.textContent = 'Ошибка (см. Console)';
      log('upload error:', err);
    }
  });

  // Стартовая инициализация
  document.addEventListener('DOMContentLoaded', () => {
    log('DOM loaded, инициализация');
    listPeople().catch(e => log('listPeople error', e));
  });

})();
