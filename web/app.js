import { Model, P, PARAM_DEFS, clamp, lerp } from './rig.js';

const canvas = document.getElementById('stage');
const statusEl = document.getElementById('status');
let model;
try {
  model = await Model.load(canvas);
  statusEl.textContent = '';
} catch (e) {
  statusEl.textContent = '読み込みに失敗しました: ' + e.message;
  throw e;
}

// ------------------------------------------------------------------ state
const target = { angleX: 0, angleY: 0, angleZ: 0, eyeX: 0, eyeY: 0 };
const state = {
  mode: 'mouse',            // mouse | camera
  manual: false,
  pointer: null, lastPointer: 0,
  blinkT: 0, nextBlink: 2, blinkDur: 0.17,
  happyUntil: 0, winkAt: -1e9,
  speech: 0,                // smoothed mouth activity, drives nodding while talking
  talk: null,               // null | 'demo' | 'mic'
  mic: null, cam: null,
  face: null,               // latest camera-derived values
};

// ------------------------------------------------------------------ input
canvas.addEventListener('pointermove', (e) => { state.pointer = [e.clientX, e.clientY]; state.lastPointer = performance.now(); });
canvas.addEventListener('pointerleave', () => { state.pointer = null; });

let drag = null;
canvas.addEventListener('pointerdown', (e) => {
  drag = { x: e.clientX, y: e.clientY, moved: false };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointerup', (e) => {
  if (drag && !drag.moved) {
    const [ix, iy] = model.screenToImage(e.clientX, e.clientY);
    if (Math.hypot(ix - 1020, iy - 650) < 420) makeHappy();
  }
  drag = null;
});
canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < 6) return;
  drag.moved = true; model.pan(dx, dy); drag.x = e.clientX; drag.y = e.clientY;
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  model.zoom(Math.exp(e.deltaY * 0.0012), e.clientX, e.clientY);
}, { passive: false });

function makeHappy() {
  state.happyUntil = performance.now() + 2600;
  spawnHearts();
}

function spawnHearts() {
  const [hx, hy] = model.imageToScreen(1020, 560);
  const scale = canvas.clientHeight / model.view.h;
  for (let i = 0; i < 4; i++) {
    const el = document.createElement('div');
    el.className = 'heart';
    const side = i % 2 ? 1 : -1;
    el.style.left = `${hx + side * (260 + Math.random() * 160) * scale}px`;
    el.style.top = `${hy - (Math.random() * 260) * scale}px`;
    el.style.setProperty('--s', (0.6 + Math.random() * 0.6) * Math.max(scale * 3, 0.5));
    el.style.animationDelay = `${i * 0.12}s`;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 2400);
  }
}

// ------------------------------------------------------------------ UI
const $ = (id) => document.getElementById(id);
$('btn-frame').onclick = () => {
  const bust = $('btn-frame').dataset.mode !== 'bust';
  $('btn-frame').dataset.mode = bust ? 'bust' : 'full';
  $('btn-frame').textContent = bust ? '全身' : 'バストアップ';
  model.setFraming(bust ? 'bust' : 'full');
};
$('btn-happy').onclick = makeHappy;
$('btn-wink').onclick = () => { state.winkAt = performance.now(); };
$('btn-talk').onclick = () => {
  if (state.talk === 'demo') { state.talk = null; $('btn-talk').classList.remove('on'); return; }
  stopMic(); state.talk = 'demo'; $('btn-talk').classList.add('on');
};
$('btn-mic').onclick = async () => {
  if (state.talk === 'mic') { stopMic(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ctx = new AudioContext();
    const an = ctx.createAnalyser(); an.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(an);
    state.mic = { stream, ctx, an, buf: new Float32Array(an.fftSize), level: 0 };
    state.talk = 'mic';
    $('btn-mic').classList.add('on'); $('btn-talk').classList.remove('on');
  } catch (e) { flash('マイクを使えませんでした: ' + e.message); }
};
function stopMic() {
  if (state.mic) { state.mic.stream.getTracks().forEach((t) => t.stop()); state.mic.ctx.close(); state.mic = null; }
  if (state.talk === 'mic') state.talk = null;
  $('btn-mic').classList.remove('on');
}
$('btn-cam').onclick = async () => {
  if (state.mode === 'camera') { stopCam(); return; }
  $('btn-cam').textContent = '準備中…';
  try { await startCam(); state.mode = 'camera'; $('btn-cam').classList.add('on'); $('btn-cam').textContent = 'カメラ追従 ON'; }
  catch (e) { flash('カメラ追従を開始できませんでした: ' + e.message); $('btn-cam').textContent = 'カメラ追従'; }
};
$('btn-panel').onclick = () => document.body.classList.toggle('panel-open');
$('manual').onchange = (e) => { state.manual = e.target.checked; };
$('bg').oninput = (e) => { document.body.dataset.bg = e.target.value; };

const sliders = {};
for (const [key, label, min, max] of PARAM_DEFS) {
  const row = document.createElement('label');
  row.className = 'row';
  row.innerHTML = `<span>${label}</span><input type="range" min="${min}" max="${max}" step="0.01" value="${P[key]}"><output></output>`;
  const input = row.querySelector('input');
  input.addEventListener('input', () => {
    if (!state.manual) { state.manual = true; $('manual').checked = true; }
    P[key] = parseFloat(input.value);
  });
  sliders[key] = { input, out: row.querySelector('output') };
  $('params').appendChild(row);
}

// ?p=angleX:1,happy:1  -> start in manual mode with those values (handy for checking poses)
const qp = new URLSearchParams(location.search).get('p');
if (qp) {
  state.manual = true; $('manual').checked = true;
  for (const kv of qp.split(',')) { const [k, v] = kv.split(':'); if (k in P) P[k] = parseFloat(v); }
}
if (new URLSearchParams(location.search).get('frame') === 'bust') $('btn-frame').click();

function flash(msg) {
  statusEl.textContent = msg;
  setTimeout(() => { if (statusEl.textContent === msg) statusEl.textContent = ''; }, 4000);
}

// ------------------------------------------------------------------ camera tracking (MediaPipe)
async function startCam() {
  const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs');
  const files = await vision.FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
  const landmarker = await vision.FaceLandmarker.createFromOptions(files, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO', numFaces: 1,
    outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
  });
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
  const video = $('cam');
  video.srcObject = stream; await video.play();
  document.body.classList.add('cam-on');
  state.cam = { landmarker, stream, video, last: -1 };
}
function stopCam() {
  if (state.cam) { state.cam.stream.getTracks().forEach((t) => t.stop()); state.cam.landmarker.close(); state.cam = null; }
  state.mode = 'mouse'; state.face = null;
  document.body.classList.remove('cam-on');
  $('btn-cam').classList.remove('on'); $('btn-cam').textContent = 'カメラ追従';
}
function pollCam() {
  const c = state.cam;
  if (!c || c.video.readyState < 2 || c.video.currentTime === c.last) return;
  c.last = c.video.currentTime;
  const r = c.landmarker.detectForVideo(c.video, performance.now());
  if (!r.faceBlendshapes || !r.faceBlendshapes.length) return;
  const bs = {};
  for (const cat of r.faceBlendshapes[0].categories) bs[cat.categoryName] = cat.score;
  const m = r.facialTransformationMatrixes[0].data; // column-major 4x4
  const R = (row, col) => m[col * 4 + row];
  const yaw = Math.asin(clamp(-R(2, 0), -1, 1));
  const pitch = Math.atan2(R(2, 1), R(2, 2));
  const roll = Math.atan2(R(1, 0), R(0, 0));
  state.face = {
    // mirrored, like looking into a mirror
    angleX: clamp(-yaw / 0.45, -1, 1),
    angleY: clamp(pitch / 0.35, -1, 1),
    angleZ: clamp(-roll / 0.35, -1, 1),
    eyeL: 1 - clamp((bs.eyeBlinkRight - 0.15) / 0.5, 0, 1),
    eyeR: 1 - clamp((bs.eyeBlinkLeft - 0.15) / 0.5, 0, 1),
    eyeX: clamp(((bs.eyeLookOutRight || 0) - (bs.eyeLookInRight || 0)) * 1.6, -1, 1),
    eyeY: clamp(((bs.eyeLookDownLeft || 0) - (bs.eyeLookUpLeft || 0)) * 1.6, -1, 1),
    mouth: clamp((bs.jawOpen - 0.05) / 0.45, 0, 1),
    smile: clamp(((bs.mouthSmileLeft + bs.mouthSmileRight) / 2 - 0.55) / 0.3, 0, 1),
  };
}

// ------------------------------------------------------------------ loop
let last = performance.now();
let talkPhase = 0, talkTarget = 0, talkForm = 0;

function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  const t = now / 1000;

  if (!state.manual) {
    pollCam();
    const f = state.face;
    if (state.mode === 'camera' && f) {
      Object.assign(target, { angleX: f.angleX, angleY: f.angleY, angleZ: f.angleZ, eyeX: f.eyeX, eyeY: f.eyeY });
    } else if (state.pointer && now - state.lastPointer < 4000) {
      const [hx, hy] = model.imageToScreen(1020, 760);
      const nx = clamp((state.pointer[0] - hx) / (canvas.clientWidth * 0.45), -1, 1);
      const ny = clamp((state.pointer[1] - hy) / (canvas.clientHeight * 0.45), -1, 1);
      Object.assign(target, { angleX: nx, angleY: -ny, angleZ: -nx * 0.25, eyeX: nx, eyeY: ny * 0.8 });
    } else {
      // idle: slow wandering gaze
      const n = (a, b, c) => Math.sin(t * a) * 0.6 + Math.sin(t * b + 1.7) * 0.4 + Math.sin(t * c + 4.1) * 0.2;
      Object.assign(target, { angleX: n(0.31, 0.53, 0.97) * 0.45, angleY: n(0.23, 0.41, 0.77) * 0.3, angleZ: n(0.19, 0.37, 0.6) * 0.4,
        eyeX: n(0.43, 0.71, 1.3) * 0.5, eyeY: n(0.29, 0.61, 1.1) * 0.3 });
    }
    // talking: small nods and sways that follow how much she is speaking
    state.speech = lerp(state.speech, P.mouthOpen, 1 - Math.exp(-dt * 3));
    if (state.talk && state.mode !== 'camera') {
      const sp = state.speech;
      target.angleY += sp * (0.38 * Math.sin(t * 4.7) + 0.1);
      target.angleX += sp * 0.2 * Math.sin(t * 1.9 + 1);
      target.angleZ += sp * 0.18 * Math.sin(t * 2.6);
    }
    // wink: tilt the head toward the closing eye
    const winkP = (now - state.winkAt) / 1300;
    const wink = winkP >= 0 && winkP <= 1 ? Math.sin(Math.PI * Math.min(1, winkP * 1.6)) ** 0.5 * (winkP < 0.6 ? 1 : 1 - (winkP - 0.6) / 0.4) : 0;
    target.angleZ += wink * 0.5;
    const e = 1 - Math.exp(-dt * (state.mode === 'camera' ? 14 : 6));
    for (const k in target) P[k] = lerp(P[k], target[k], k.startsWith('eye') ? Math.min(1, e * 2) : e);
    // the body follows the head with a delay and smaller amplitude
    const eb = 1 - Math.exp(-dt * (state.mode === 'camera' ? 5 : 2.5));
    // (the upper body also leans toward where she is looking)
    P.bodyX = lerp(P.bodyX, P.angleX * 0.8, eb);
    P.bodyY = lerp(P.bodyY, P.angleY * 0.6, eb);
    P.bodyZ = lerp(P.bodyZ, clamp(P.angleX * 0.55 + P.angleZ * 0.5, -1, 1), eb);
    P.breath = (Math.sin(t * 1.9) + 1) / 2;

    // eyes
    if (state.mode === 'camera' && f) {
      P.eyeOpenL = lerp(P.eyeOpenL, f.eyeL, 0.6); P.eyeOpenR = lerp(P.eyeOpenR, f.eyeR, 0.6);
    } else {
      state.nextBlink -= dt;
      if (state.nextBlink <= 0 && state.blinkT <= 0) {
        state.blinkT = state.blinkDur;
        state.nextBlink = 1.5 + Math.random() * 4.5;
        if (Math.random() < 0.15) state.nextBlink = 0.25; // occasional double blink
      }
      let open = 1;
      if (state.blinkT > 0) {
        state.blinkT -= dt;
        const p = 1 - state.blinkT / state.blinkDur;
        open = p < 0.45 ? 1 - p / 0.45 : (p < 0.55 ? 0 : (p - 0.55) / 0.45);
      }
      P.eyeOpenL = P.eyeOpenR = clamp(open, 0, 1);
    }
    if (wink > 0) P.eyeOpenR = Math.min(P.eyeOpenR, 1 - clamp(wink * 1.4, 0, 1));

    // mouth
    let mouth = 0;
    if (state.mode === 'camera' && f) mouth = f.mouth;
    if (state.talk === 'demo') {
      talkPhase -= dt;
      if (talkPhase <= 0) {
        const pause = Math.random() < 0.12;
        talkPhase = pause ? 0.25 + Math.random() * 0.3 : 0.07 + Math.random() * 0.1;
        talkTarget = pause ? 0 : 0.25 + Math.random() * 0.75;
        talkForm = pause ? 0 : Math.random() * 2 - 1; // -1 round (o/u) .. +1 wide (i/e)
      }
      mouth = Math.max(mouth, talkTarget);
    } else if (state.talk === 'mic' && state.mic) {
      const { an, buf } = state.mic;
      an.getFloatTimeDomainData(buf);
      let s = 0; for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
      const rms = Math.sqrt(s / buf.length);
      state.mic.level = lerp(state.mic.level, clamp((rms - 0.012) * 9, 0, 1), 0.5);
      mouth = Math.max(mouth, state.mic.level);
    }
    P.mouthOpen = lerp(P.mouthOpen, mouth, 1 - Math.exp(-dt * 25));
    P.mouthForm = lerp(P.mouthForm, state.talk === 'demo' ? talkForm : 0, 1 - Math.exp(-dt * 18));

    const wantHappy = now < state.happyUntil || (state.mode === 'camera' && f && f.smile > 0.5) ? 1 : 0;
    P.happy = lerp(P.happy, wantHappy, 1 - Math.exp(-dt * 14));
    if (P.happy < 0.02) P.happy = 0;
  }

  model.update(dt);
  model.draw();

  for (const k in sliders) {
    const s = sliders[k];
    if (!state.manual) s.input.value = P[k];
    s.out.textContent = P[k].toFixed(2);
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
window.__mafuyu = { P, model, state };
