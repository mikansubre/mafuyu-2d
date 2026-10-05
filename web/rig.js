// Live2D-style mesh deformation rig for a single flat illustration.
// All geometry lives in source-image pixel space (4_4.png, 2000x4156).

const ASSET = (name) => (window.MAFUYU_ASSETS && window.MAFUYU_ASSETS[name]) || `assets/${name}`;

// ------------------------------------------------------------------ params
const P = {
  angleX: 0, angleY: 0, angleZ: 0,     // head yaw / pitch / roll  (-1..1)
  bodyX: 0, bodyY: 0, bodyZ: 0,        // body turn / bend / lean   (-1..1)
  eyeX: 0, eyeY: 0,                    // iris                      (-1..1)
  eyeOpenL: 1, eyeOpenR: 1,            // 0 closed .. 1 open
  mouthOpen: 0,                        // 0..1
  mouthForm: 0,                        // -1 round .. +1 wide
  happy: 0,                            // ^ ^ expression            (0..1)
  breath: 0,                           // 0..1
};
const PARAM_DEFS = [
  ['angleX', '顔 左右', -1, 1], ['angleY', '顔 上下', -1, 1], ['angleZ', '顔 傾き', -1, 1],
  ['bodyX', '体 左右', -1, 1], ['bodyY', '体 上下', -1, 1], ['bodyZ', '体 傾き', -1, 1],
  ['eyeX', '目玉 X', -1, 1], ['eyeY', '目玉 Y', -1, 1],
  ['eyeOpenL', '左目 開閉', 0, 1], ['eyeOpenR', '右目 開閉', 0, 1],
  ['mouthOpen', '口 開閉', 0, 1], ['mouthForm', '口 形', -1, 1], ['happy', '笑顔', 0, 1], ['breath', '呼吸', 0, 1],
];

// rig constants (image px)
// yaw/pitch: extra px the face centre moves; shiftX/Y: px the whole head slides
// pivot: roll centre, placed near the jaw so tilting barely moves the neck
const HEAD = { cx: 1020, cy: 680, rx: 600, ry: 420, yaw: 34, pitch: 20, shiftX: 10, shiftY: 6, roll: 0.07, pivot: [1020, 900] };
// below the chin the head displacement fades linearly, so the neck shears in straight lines instead of bending
const NECK = { top: 945, bottom: 1060, cut: 1000, follow: 0.35 };
const AHOGE = { bx: 1035, by: 292 };
const TAIL_ANCHOR = { L: [705, 945], R: [1335, 945] };
// upper body turns / bends / leans from the hips; weight grows linearly from hipY (0) to topY (1)
const BODY = { cx: 1020, hipY: 2250, topY: 1150, rx: 520, yaw: 45, shiftX: 42, shiftY: 20, roll: 0.09 };

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;

// ------------------------------------------------------------------ physics
class Spring {
  constructor(k, c) { this.k = k; this.c = c; this.x = 0; this.v = 0; }
  step(force, dt) {
    const a = -this.k * this.x - this.c * this.v + force;
    this.v += a * dt; this.x += this.v * dt;
    return this.x;
  }
}
const phys = {
  tailL: new Spring(28, 4.2), tailR: new Spring(26, 4.0),
  ahoge: new Spring(60, 3.0), skirt: new Spring(22, 3.4),
  lockX: new Spring(40, 5.0), lockY: new Spring(40, 5.0),
};

// frame-level derived values used by deform()
const F = {
  ca: 0, sa: 0, cr: 0, sr: 0, ay: 0, ax: 0, az: 0,
  tailL: 0, tailR: 0, ahoge: 0, skirt: 0, skirtFlare: 0, lockX: 0, lockY: 0,
  bodyX: 0, bodyY: 0, bodyRot: 0, breath: 0,
  anchorL: [0, 0], anchorR: [0, 0],
};

function headDisp(x, y, out) {
  // pseudo-3D turn. Yaw shift depends on x only and pitch shift on y only (like turning a cylinder),
  // so vertical strands and the face outline stay straight instead of bending.
  // The centre column moves most, the sides least; (1-u^2)^2 keeps the slope bounded.
  const u = (x - HEAD.cx) / HEAD.rx, v = (y - HEAD.cy) / HEAD.ry;
  const bx = Math.max(0, 1 - u * u) ** 2, by = Math.max(0, 1 - v * v) ** 2;
  let dx = F.ax * (HEAD.shiftX + HEAD.yaw * bx);
  let dy = F.ay * (HEAD.shiftY + HEAD.pitch * by);
  // roll around the jaw
  const rx = x + dx - HEAD.pivot[0], ry = y + dy - HEAD.pivot[1];
  dx += rx * F.cr - ry * F.sr - rx;
  dy += rx * F.sr + ry * F.cr - ry;
  out[0] = dx; out[1] = dy;
}

// ------------------------------------------------------------------ learned head turn (THA3 warp fields)
// flow.bin holds forward displacement fields baked by ai/export_flow.py for a yaw x pitch grid.
// Each frame the four nearest poses are blended into one field; every layer samples it at its rest position.
const FLOW = { on: false, gainYaw: 0.9, gainPitch: 0.55 };
function initFlow(meta, buf) {
  const m = meta.flow;
  FLOW.m = m; FLOW.data = new Int16Array(buf);
  FLOW.n = m.nx * m.ny;
  FLOW.fx = new Float32Array(FLOW.n); FLOW.fy = new Float32Array(FLOW.n);
  FLOW.on = true;
}
function bracket(arr, v) {
  v = clamp(v, arr[0], arr[arr.length - 1]);
  let i = 0;
  while (i < arr.length - 2 && v > arr[i + 1]) i++;
  return [i, (v - arr[i]) / (arr[i + 1] - arr[i])];
}
function blendFlow(yaw, pitch) {
  const m = FLOW.m, n = FLOW.n, d = FLOW.data, u = m.unit;
  const [i, ti] = bracket(m.yaws, yaw), [j, tj] = bracket(m.pitches, pitch);
  const NY = m.yaws.length;
  const w = [[(1 - ti) * (1 - tj), j, i], [ti * (1 - tj), j, i + 1], [(1 - ti) * tj, j + 1, i], [ti * tj, j + 1, i + 1]];
  FLOW.fx.fill(0); FLOW.fy.fill(0);
  for (const [wk, pj, pi] of w) {
    if (wk < 1e-4) continue;
    const base = (pj * NY + pi) * 2 * n, k = wk * u;
    for (let q = 0; q < n; q++) { FLOW.fx[q] += d[base + q] * k; FLOW.fy[q] += d[base + n + q] * k; }
  }
}
function flowAt(x, y, out) {
  const m = FLOW.m;
  const gx = (x - m.x0) / m.step, gy = (y - m.y0) / m.step;
  if (gx < 0 || gy < 0 || gx >= m.nx - 1 || gy >= m.ny - 1) { out[0] = 0; out[1] = 0; return; }
  const ix = gx | 0, iy = gy | 0, tx = gx - ix, ty = gy - iy;
  const a = iy * m.nx + ix, b = a + 1, c = a + m.nx, e = c + 1;
  out[0] = (FLOW.fx[a] * (1 - tx) + FLOW.fx[b] * tx) * (1 - ty) + (FLOW.fx[c] * (1 - tx) + FLOW.fx[e] * tx) * ty;
  out[1] = (FLOW.fy[a] * (1 - tx) + FLOW.fy[b] * tx) * (1 - ty) + (FLOW.fy[c] * (1 - tx) + FLOW.fy[e] * tx) * ty;
}

// twin tails + ribbons hang from the hair ties: outside the face, the learned field is sampled at the tie,
// so ribbon and tail move together with the head instead of being squashed/torn apart.
// Below the head/back cut the follow fades toward the tail tips (the spring swing adds on top).
const TIES = { L: [738, 452], R: [1304, 452] };
function tieZone(x, y) {
  if (y < 400) return null;
  const wy = smooth(400, 440, y);
  // wide (100px) ramps: the tie's motion differs a lot from the face's, a narrow ramp tears the side hair
  if (x < 860) return [TIES.L, smooth(860, 760, x) * wy];
  if (x > 1190) return [TIES.R, smooth(1190, 1290, x) * wy];
  return null;
}
const _hd = [0, 0], _fl = [0, 0];
// kind: 'head' (hair, face, eyes, mouth), 'body' (uniform, neck, limbs), 'back' (lower twin tails)
function deform(x, y, wTail, wSkirt, wSway, kind, out) {
  let px = x, py = y;
  // legs stand still (they sit behind the skirt and below the hips)
  if (kind === 'legs') { out[0] = x; out[1] = y; return; }
  // the learned head field only matters for the head, the twin tails and the neck; elsewhere on the body
  // it is noise (it shifted the hair resting on the shoulders)
  const wFlow = kind === 'body' ? (1 - smooth(110, 190, Math.abs(x - HEAD.cx))) * smooth(1150, 1000, y) : 1;
  if (FLOW.on && wFlow > 0) {
    // the field changes sharply across the jaw; below the chin the head layer reuses the chin row
    // (only around the chin: elsewhere, e.g. the twin tails, the head layer must match the back layer at the cut)
    const nearChin = kind === 'head' && Math.abs(x - HEAD.cx) < 200;
    let sx = x, sy = nearChin ? Math.min(y, NECK.top - 10) : y;
    const tz = kind === 'head' || kind === 'back' ? tieZone(x, y) : null;
    if (tz) { sx = lerp(sx, tz[0][0], tz[1]); sy = lerp(sy, tz[0][1], tz[1]); }
    flowAt(sx, sy, _fl);
    const follow = kind === 'back' ? 1 - 0.5 * clamp((y - NECK.cut) / 800, 0, 1) : 1;
    px += _fl[0] * follow * wFlow; py += _fl[1] * follow * wFlow;
    // ribbons / tails swing a little from the tie, following the side-lock spring
    if (tz) px += tz[1] * F.lockX * 0.8 * clamp((y - tz[0][1]) / 500, 0, 1);
  }

  if (kind === 'head') {
    // ahoge
    if (y < 305 && y > 140 && x > 985 && x < 1215) {
      const d = Math.hypot(x - AHOGE.bx, y - AHOGE.by);
      const w = clamp(d / 170, 0, 1) * smooth(305, 285, y);
      const a = F.ahoge * w;
      const dx = px - AHOGE.bx, dy = py - AHOGE.by;
      px = AHOGE.bx + dx * Math.cos(a) - dy * Math.sin(a);
      py = AHOGE.by + dx * Math.sin(a) + dy * Math.cos(a);
    }
    // face-framing side locks lag behind the head (spring), more toward the tips
    if (wSway > 0) { px += wSway * F.lockX; py += wSway * F.lockY; }
    // the head is its own layer: it moves as a whole and slides over the neck
    headDisp(px, Math.min(py, NECK.top), _hd);
    px += _hd[0]; py += _hd[1];
  } else if (kind === 'body') {
    // the neck (painted up behind the jaw) follows the chin, fading out toward the collar
    if (y < NECK.bottom) {
      const wy = y <= NECK.top ? 1 : 1 - (y - NECK.top) / (NECK.bottom - NECK.top);
      const wx = 1 - smooth(100, 190, Math.abs(x - HEAD.cx));
      headDisp(HEAD.cx, NECK.top, _hd);
      px += _hd[0] * wy * wx * NECK.follow; py += _hd[1] * wy * wx * NECK.follow;
    }
  } else if (kind === 'back') {
    // twin tails: the top edge matches the head layer, then the spring swing takes over
    const left = x < HEAD.cx;
    const t = clamp((y - NECK.cut) / 800, 0, 1);
    headDisp(px, NECK.top, _hd);
    const sw = left ? F.tailL : F.tailR;
    px += _hd[0] * (1 - 0.6 * t) + wTail * sw * Math.pow(t, 1.6);
    py += _hd[1] * (1 - 0.6 * t) - wTail * Math.abs(sw) * 0.08 * t * t;
  }

  // breathing (shoulders rise, chest widens a touch)
  const wUp = smooth(2150, 1250, y);
  py -= F.breath * 5 * wUp;
  px += (x - 1020) * F.breath * 0.006 * smooth(1000, 1150, y) * smooth(1800, 1450, y);

  // skirt
  if (wSkirt > 0) {
    px += wSkirt * (F.skirt + (x - 1040) * F.skirtFlare);
    py -= wSkirt * Math.abs(F.skirt) * 0.15;
  }

  // upper body: turn (chest centre moves more than the shoulders), bend, then lean around the hips.
  // Everything above the hips (head layers included) rides along, so the head sits on a moving body.
  const g = clamp((BODY.hipY - y) / (BODY.hipY - BODY.topY), 0, 1);
  if (g > 0) {
    const u = (x - BODY.cx) / BODY.rx;
    const b = Math.max(0, 1 - u * u) ** 2;
    px += F.bodyX * (BODY.shiftX + BODY.yaw * b) * g;
    py -= F.bodyY * BODY.shiftY * g;
    const a = F.bodyRot * g;
    const dx = px - BODY.cx, dy = py - BODY.hipY;
    px = BODY.cx + dx * Math.cos(a) - dy * Math.sin(a);
    py = BODY.hipY + dx * Math.sin(a) + dy * Math.cos(a);
  }
  out[0] = px; out[1] = py;
}

// ------------------------------------------------------------------ GL helpers
const VS = `#version 300 es
in vec2 aPos; in vec2 aUv;
uniform vec4 uView; // scale.xy, offset.xy (image px -> clip)
out vec2 vUv;
void main(){ vUv = aUv; gl_Position = vec4(aPos * uView.xy + uView.zw, 0.0, 1.0); }`;
const FS = `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uTex; uniform sampler2D uMask;
uniform bool uUseMask; uniform vec2 uUvOff; uniform float uAlpha;
out vec4 o;
void main(){
  vec2 uv = vUv - uUvOff;
  vec4 c = texture(uTex, uv);
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) c = vec4(0.0);
  if (uUseMask) c *= texture(uMask, vUv).a;
  o = c * uAlpha;
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function loadImage(src) {
  return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
}

function makeTexture(gl, img) {
  const max = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  let source = img;
  if (img.width > max || img.height > max) {
    const s = max / Math.max(img.width, img.height);
    const c = document.createElement('canvas');
    c.width = Math.floor(img.width * s); c.height = Math.floor(img.height * s);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    source = c;
  }
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

class Mesh {
  // rest: Float32Array of image-px positions, uv, weights per vertex, indices
  constructor(gl, rest, uv, wTail, wSkirt, wSway, idx, kind) {
    this.gl = gl; this.rest = rest; this.wTail = wTail; this.wSkirt = wSkirt; this.wSway = wSway; this.kind = kind;
    this.n = rest.length / 2; this.pos = new Float32Array(rest.length); this.count = idx.length;
    this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
    this.pbuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.pbuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.pos, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    const ub = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, ub);
    gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
    const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
  }
  update(local) {
    const o = [0, 0];
    for (let i = 0; i < this.n; i++) {
      let x = this.rest[i * 2], y = this.rest[i * 2 + 1];
      if (local) { const l = local(x, y); x = l[0]; y = l[1]; }
      deform(x, y, this.wTail ? this.wTail[i] : 0, this.wSkirt ? this.wSkirt[i] : 0, this.wSway ? this.wSway[i] : 0, this.kind, o);
      this.pos[i * 2] = o[0]; this.pos[i * 2 + 1] = o[1];
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pbuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.pos);
  }
  draw() {
    const gl = this.gl; gl.bindVertexArray(this.vao);
    gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_INT, 0);
  }
}

function gridMesh(gl, kind, x0, y0, w, h, step, keepCell, sampleW) {
  const nx = Math.max(1, Math.ceil(w / step)), ny = Math.max(1, Math.ceil(h / step));
  const rest = [], uv = [], wt = [], ws = [], wsw = [], idx = [];
  for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
    const u = i / nx, v = j / ny, x = x0 + u * w, y = y0 + v * h;
    rest.push(x, y); uv.push(u, v);
    if (sampleW) { const s = sampleW(x, y); wt.push(s[0]); ws.push(s[1]); wsw.push(s[2]); }
  }
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    if (keepCell && !keepCell(x0 + (i + 0.5) * w / nx, y0 + (j + 0.5) * h / ny)) continue;
    const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 1, d = c + 1;
    idx.push(a, b, c, b, d, c);
  }
  return new Mesh(gl, new Float32Array(rest), new Float32Array(uv),
    sampleW ? new Float32Array(wt) : null, sampleW ? new Float32Array(ws) : null,
    sampleW && kind === 'head' ? new Float32Array(wsw) : null, new Uint32Array(idx), kind);
}

// ------------------------------------------------------------------ model
export class Model {
  static async load(canvas) {
    const m = new Model();
    await m.init(canvas);
    return m;
  }

  async init(canvas) {
    const gl = canvas.getContext('webgl2', { premultipliedAlpha: true, alpha: true, antialias: true });
    if (!gl) throw new Error('WebGL2 が使えません');
    this.gl = gl; this.canvas = canvas;
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.bindAttribLocation(prog, 0, 'aPos'); gl.bindAttribLocation(prog, 1, 'aUv');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    this.prog = prog;
    this.u = {};
    for (const n of ['uView', 'uTex', 'uMask', 'uUseMask', 'uUvOff', 'uAlpha']) this.u[n] = gl.getUniformLocation(prog, n);

    const meta = window.MAFUYU_PARTS || await (await fetch(ASSET('parts.json'))).json();
    this.meta = meta;
    const L = meta.layers;
    const names = Object.keys(L);
    const imgs = {};
    await Promise.all(names.map(async (n) => { imgs[n] = await loadImage(ASSET(L[n].file)); }));
    const wImg = await loadImage(ASSET(meta.weights.file));
    if (meta.flow) initFlow(meta, await (await fetch(ASSET(meta.flow.file))).arrayBuffer());
    this.tex = {};
    for (const n of names) this.tex[n] = makeTexture(gl, imgs[n]);

    // weights, sampled on the CPU once
    const ww = meta.weights.w, wh = meta.weights.h, ws = meta.weights.scale;
    const c = document.createElement('canvas'); c.width = ww; c.height = wh;
    const cx = c.getContext('2d', { willReadFrequently: true });
    cx.drawImage(wImg, 0, 0, ww, wh);
    const wData = cx.getImageData(0, 0, ww, wh).data;
    const at = (x, y) => (clamp(Math.round(y / ws), 0, wh - 1) * ww + clamp(Math.round(x / ws), 0, ww - 1)) * 4;
    const sampleW = (x, y) => { const i = at(x, y); return [wData[i] / 255, wData[i + 1] / 255, wData[i + 2] / 255]; };

    // big layers (back / body / head) get a dense mesh limited to their opaque area
    const STEP = 10;
    this.layers = [];
    for (const kind of ['back', 'legs', 'body', 'head']) {
      if (!L[kind]) continue;
      const l = L[kind];
      const lw = Math.ceil(l.w / ws), lh = Math.ceil(l.h / ws);
      const lc = document.createElement('canvas'); lc.width = lw; lc.height = lh;
      const lx = lc.getContext('2d', { willReadFrequently: true });
      lx.drawImage(imgs[kind], 0, 0, lw, lh);
      const aData = lx.getImageData(0, 0, lw, lh).data;
      const keep = (x, y) => {
        for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
          const sx = Math.round((x - l.x + dx * STEP * 0.6) / ws), sy = Math.round((y - l.y + dy * STEP * 0.6) / ws);
          if (sx >= 0 && sy >= 0 && sx < lw && sy < lh && aData[(sy * lw + sx) * 4 + 3] > 0) return true;
        }
        return false;
      };
      this.layers.push({ kind, mesh: gridMesh(gl, kind, l.x, l.y, l.w, l.h, STEP, keep, sampleW) });
    }

    this.parts = {};
    for (const n of names) {
      const l = L[n];
      if (l.mesh || n.endsWith('_mask')) continue;
      this.parts[n] = { l, mesh: gridMesh(gl, 'head', l.x, l.y, l.w, l.h, 8, null, null) };
    }
    this.view = { cx: 1020, cy: 2100, h: 4200, tcx: 1020, tcy: 2100, th: 4200 };
    this.setFraming('full', true);
  }

  setFraming(mode, instant) {
    const [x0, y0, x1, y1] = this.meta.bbox;
    const v = this.view;
    if (mode === 'bust') { v.tcx = 1020; v.tcy = 1050; v.th = 1650; }
    else { v.tcx = (x0 + x1) / 2; v.tcy = (y0 + y1) / 2 + (y1 - y0) * 0.05; v.th = (y1 - y0) * 1.16; }
    if (instant) { v.cx = v.tcx; v.cy = v.tcy; v.h = v.th; }
  }

  zoom(factor, sx, sy) {
    const v = this.view;
    const [ix, iy] = this.screenToImage(sx, sy);
    v.th = clamp(v.th * factor, 500, 6000);
    // keep point under cursor fixed (approximately, on target)
    const s = this.canvas.clientHeight / v.th;
    v.tcx = ix - (sx - this.canvas.clientWidth / 2) / s;
    v.tcy = iy - (sy - this.canvas.clientHeight / 2) / s;
  }

  pan(dx, dy) {
    const s = this.canvas.clientHeight / this.view.th;
    this.view.tcx -= dx / s; this.view.tcy -= dy / s;
  }

  screenToImage(sx, sy) {
    const v = this.view, s = this.canvas.clientHeight / v.h;
    return [v.cx + (sx - this.canvas.clientWidth / 2) / s, v.cy + (sy - this.canvas.clientHeight / 2) / s];
  }

  imageToScreen(ix, iy) {
    const v = this.view, s = this.canvas.clientHeight / v.h;
    return [(ix - v.cx) * s + this.canvas.clientWidth / 2, (iy - v.cy) * s + this.canvas.clientHeight / 2];
  }

  // ---------------------------------------------------------------- per frame
  update(dt) {
    const prevAnchorX = (F.anchorL[0] + F.anchorR[0]) / 2;
    const prevBody = F.bodyX * BODY.shiftX + F.bodyRot * 600;

    // with the learned fields the analytic yaw/pitch is off; headDisp then only does the roll
    F.ax = FLOW.on ? 0 : P.angleX; F.ay = FLOW.on ? 0 : -P.angleY; F.az = P.angleZ * HEAD.roll;
    if (FLOW.on) blendFlow(-P.angleX * FLOW.gainYaw, P.angleY * FLOW.gainPitch);
    F.cr = Math.cos(F.az); F.sr = Math.sin(F.az);
    F.breath = P.breath;
    F.bodyX = P.bodyX; F.bodyY = P.bodyY; F.bodyRot = P.bodyZ * BODY.roll;
    for (const side of ['L', 'R']) {
      const [ax, ay] = TAIL_ANCHOR[side];
      headDisp(ax, ay, _hd);
      if (FLOW.on) { flowAt(ax, 800, _fl); _hd[0] += _fl[0]; _hd[1] += _fl[1]; }
      F['anchor' + side] = [_hd[0], _hd[1]];
    }

    // physics driven by head / body velocity
    const h = Math.min(dt, 1 / 30);
    const headVel = ((F.anchorL[0] + F.anchorR[0]) / 2 - prevAnchorX) / Math.max(h, 1e-3);
    const bodyVel = (F.bodyX * BODY.shiftX + F.bodyRot * 600 - prevBody) / Math.max(h, 1e-3);
    const t = performance.now() / 1000;
    const wind = Math.sin(t * 1.3) * 4 + Math.sin(t * 2.7 + 1) * 2;
    F.tailL = phys.tailL.step(-headVel * 5 - bodyVel * 3 + wind * 12 + F.az * -600, h);
    F.tailR = phys.tailR.step(-headVel * 5 - bodyVel * 3 + wind * 11 + F.az * -600, h);
    F.ahoge = phys.ahoge.step(-headVel * 0.05 + Math.sin(t * 2.1) * 0.6 + F.az * -2, h) * 1.0;
    F.skirt = phys.skirt.step(-bodyVel * 5 + Math.sin(t * 1.1) * 8, h);
    // side locks: inertia from head turn / nod / tilt speed (parameter units per second)
    const prev = this._prevPose || [P.angleX, P.angleY, P.angleZ];
    const vx = (P.angleX - prev[0]) / Math.max(h, 1e-3), vy = (P.angleY - prev[1]) / Math.max(h, 1e-3);
    const vz = (P.angleZ - prev[2]) / Math.max(h, 1e-3);
    this._prevPose = [P.angleX, P.angleY, P.angleZ];
    F.lockX = clamp(phys.lockX.step(-vx * 260 - vz * 200 + wind * 6, h), -22, 22);
    F.lockY = clamp(phys.lockY.step(vy * 120, h), -10, 10);
    F.skirtFlare = Math.abs(F.skirt) * 0.0006;
    const lim = (v, m) => clamp(v, -m, m);
    F.tailL = lim(F.tailL, 90); F.tailR = lim(F.tailR, 90); F.ahoge = lim(F.ahoge, 0.25); F.skirt = lim(F.skirt, 30);

    for (const ly of this.layers) ly.mesh.update(null);

    const eyeLocal = (side, open) => {
      const py = this.parts[`eye${side}_white`].l.pivotY;
      const k = Math.max(open, 0.02);
      return (x, y) => [x, py + (y - py) * k];
    };
    for (const side of ['L', 'R']) {
      const open = side === 'L' ? P.eyeOpenL : P.eyeOpenR;
      const f = eyeLocal(side, open);
      for (const n of ['white', 'iris', 'line']) this.parts[`eye${side}_${n}`].mesh.update(f);
      this.parts[`eye${side}_closed`].mesh.update(null);
      this.parts[`eye${side}_happy`].mesh.update(null);
    }
    this.parts.mouth_smile.mesh.update(null);
    const mo = this.parts.mouth_open.l;
    const pivot = mo.y + mo.h * 0.2;
    const mk = 0.15 + 0.85 * clamp(Math.max(P.mouthOpen, P.happy * 0.9), 0, 1);
    // mouthForm: -1 round (o/u: narrower, taller) .. +1 wide (i/e: wider, flatter)
    const sx = (1 + (mk - 1) * 0.25) * (1 + 0.22 * P.mouthForm), sy = mk * (1 - 0.25 * P.mouthForm);
    const mcx = mo.x + mo.w / 2;
    this.parts.mouth_open.mesh.update((x, y) => [mcx + (x - mcx) * sx, pivot + (y - pivot) * sy]);

    const v = this.view, e = 1 - Math.exp(-dt * 8);
    v.cx = lerp(v.cx, v.tcx, e); v.cy = lerp(v.cy, v.tcy, e); v.h = lerp(v.h, v.th, e);
  }

  draw() {
    const gl = this.gl, c = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.round(c.clientWidth * dpr), H = Math.round(c.clientHeight * dpr);
    if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.prog);
    const v = this.view, s = 2 / v.h, aspect = c.clientWidth / c.clientHeight;
    gl.uniform4f(this.u.uView, s / aspect, -s, -v.cx * s / aspect, v.cy * s);
    gl.uniform1i(this.u.uTex, 0); gl.uniform1i(this.u.uMask, 1);

    const drawPart = (mesh, tex, alpha, mask, uvOff) => {
      if (alpha <= 0.001) return;
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(this.u.uUseMask, mask ? 1 : 0);
      if (mask) { gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, mask); }
      gl.uniform2f(this.u.uUvOff, uvOff ? uvOff[0] : 0, uvOff ? uvOff[1] : 0);
      gl.uniform1f(this.u.uAlpha, alpha);
      mesh.draw();
    };

    for (const ly of this.layers) drawPart(ly.mesh, this.tex[ly.kind], 1);
    const happy = clamp(P.happy, 0, 1);
    for (const side of ['L', 'R']) {
      const open = side === 'L' ? P.eyeOpenL : P.eyeOpenR;
      const vis = smooth(0.08, 0.3, open) * (1 - happy);
      const wl = this.parts[`eye${side}_white`].l;
      const off = [P.eyeX * 6 / wl.w, P.eyeY * 4 / wl.h];
      drawPart(this.parts[`eye${side}_white`].mesh, this.tex[`eye${side}_white`], vis);
      drawPart(this.parts[`eye${side}_iris`].mesh, this.tex[`eye${side}_iris`], vis, this.tex[`eye${side}_mask`], off);
      drawPart(this.parts[`eye${side}_line`].mesh, this.tex[`eye${side}_line`], vis);
      drawPart(this.parts[`eye${side}_closed`].mesh, this.tex[`eye${side}_closed`], (1 - smooth(0.12, 0.35, open)) * (1 - happy));
      drawPart(this.parts[`eye${side}_happy`].mesh, this.tex[`eye${side}_happy`], happy);
    }
    const mOpen = clamp(Math.max(P.mouthOpen, happy * 0.9), 0, 1);
    drawPart(this.parts.mouth_smile.mesh, this.tex.mouth_smile, 1 - smooth(0.04, 0.2, mOpen));
    drawPart(this.parts.mouth_open.mesh, this.tex.mouth_open, smooth(0.04, 0.2, mOpen));
  }
}

export { P, PARAM_DEFS, FLOW, clamp, lerp, smooth };
