"""Self-contained WebGL viewer of the simulated voxel structure.

Extracts the solid/air boundary faces of each scene part from the occupancy
grid and writes a single HTML file (no external assets) with an orbit/zoom
viewer. What is shown is exactly what the FDTD kernel sees — the voxelized
parts, the receiver probes, and the driver disc.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from headphone_sims.geometry.scene import BuiltScene

_PART_STYLE: dict[str, tuple[str, str]] = {
    "driver": ("振動板(ドーム)", "#5B8DD9"),
    "baffle": ("バッフル壁", "#93A3B2"),
    "cup": ("イヤカップ", "#7D6B5A"),
    "filter_1": ("フィルタ 1", "#E08A4C"),
    "filter_2": ("フィルタ 2", "#D6B24C"),
    "filter_3": ("フィルタ 3", "#B8874C"),
    "pinna": ("ピンナ(耳介)", "#C98D6B"),
    "head": ("頭部表面", "#6E7F8D"),
}
_FALLBACK_COLORS = ["#8E9BAA", "#C4A25E", "#9A7FB1"]


def _extract_faces(built: BuiltScene) -> tuple[list[dict[str, object]], int]:
    """Packed boundary faces per part: uint16 [axis*2+sign, plane, a, b]."""
    grid = built.grid
    ids = np.zeros(grid.shape, dtype=np.uint8)
    names = []
    for pid, (name, occ) in enumerate(built.parts, start=1):
        mask = occ.numpy() & (ids == 0)
        ids[mask] = pid
        names.append(name)
    solid = ids > 0

    per_part: dict[int, list[np.ndarray]] = {p: [] for p in range(1, len(names) + 1)}
    for axis in range(3):
        lo = solid.take(range(0, solid.shape[axis] - 1), axis=axis)
        hi = solid.take(range(1, solid.shape[axis]), axis=axis)
        trans = lo != hi
        idx = np.argwhere(trans)
        if not len(idx):
            continue
        lower_solid = lo[trans]  # True: solid below the face -> normal +axis
        plane = idx[:, axis] + 1
        others = [a for a in range(3) if a != axis]
        owner_idx = idx.copy()
        owner_idx[:, axis] += (~lower_solid).astype(int)
        owner = ids[owner_idx[:, 0], owner_idx[:, 1], owner_idx[:, 2]]
        packed = np.stack(
            [
                (axis * 2 + (~lower_solid).astype(int)).astype(np.uint16),
                plane.astype(np.uint16),
                idx[:, others[0]].astype(np.uint16),
                idx[:, others[1]].astype(np.uint16),
            ],
            axis=1,
        )
        for pid in range(1, len(names) + 1):
            sel = owner == pid
            if sel.any():
                per_part[pid].append(packed[sel])

    parts_js: list[dict[str, object]] = []
    total = 0
    for pid, name in enumerate(names, start=1):
        if not per_part[pid]:
            continue
        arr = np.concatenate(per_part[pid]).astype("<u2")
        total += len(arr)
        label, color = _PART_STYLE.get(name, (name, _FALLBACK_COLORS[pid % len(_FALLBACK_COLORS)]))
        parts_js.append(
            {
                "name": name,
                "label": label,
                "color": color,
                "count": len(arr),
                "data": base64.b64encode(arr.tobytes()).decode(),
            }
        )
    return parts_js, total


def write_structure_viewer(
    built: BuiltScene,
    out_path: str | Path,
    title: str = "ヘッドホンシーン3D",
    driver_radius_m: float | None = None,
) -> Path:
    parts_js, total = _extract_faces(built)
    grid = built.grid
    dx_mm = grid.dx * 1e3
    probes_mm = (built.probe_positions * 1e3).astype("<f4")
    meta = {
        "dxMm": dx_mm,
        "driverCenter": [c * 1e3 for c in built.driver_center],
        "driverRadiusMm": (driver_radius_m or 0.02) * 1e3,
        "extentMm": [s * dx_mm for s in grid.shape],
    }
    legend = "\n".join(
        f'<label><input type="checkbox" checked data-part="{p["name"]}">'
        f'<span class="swatch" style="background:{p["color"]}"></span>{p["label"]}</label>'
        for p in parts_js
    )
    html = _TEMPLATE
    for key, value in {
        "__TITLE__": title,
        "__PARTS__": json.dumps(parts_js),
        "__META__": json.dumps(meta),
        "__PROBES__": base64.b64encode(probes_mm.tobytes()).decode(),
        "__LEGEND__": legend,
        "__STATS__": (
            f"格子 {dx_mm:g} mm / {grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]} "
            f"= {grid.n_cells / 1e6:.1f}M セル · 境界面 {total:,} · "
            f"プローブ {len(probes_mm)} 点"
        ),
    }.items():
        html = html.replace(key, value)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


_TEMPLATE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --ground:#0F1519; --panel:#161E24; --ink:#E5EBEF; --muted:#8FA0AC;
  --accent:#E08A4C; --line:#243039; }
* { box-sizing:border-box; }
body { margin:0; background:var(--ground); color:var(--ink);
  font-family:"Hiragino Sans","Noto Sans JP","Yu Gothic UI",system-ui,sans-serif;
  font-size:15px; line-height:1.7; }
main { max-width:1080px; margin:0 auto; padding:1.4rem 1.2rem 2.5rem; }
h1 { font-size:1.3rem; font-weight:800; margin:0 0 .6rem; }
#stage { position:relative; border:1px solid var(--line); border-radius:10px;
  overflow:hidden; background:#0A0F13; }
canvas { display:block; width:100%; height:70vh; min-height:420px; cursor:grab; }
canvas:active { cursor:grabbing; }
.hint { position:absolute; left:.9rem; bottom:.7rem; font-size:.72rem;
  color:var(--muted); background:rgba(10,15,19,.72); padding:.2rem .6rem;
  border-radius:5px; pointer-events:none; }
.axis-tag { position:absolute; right:.9rem; bottom:.7rem; font-size:.72rem;
  color:var(--muted); background:rgba(10,15,19,.72); padding:.2rem .6rem; border-radius:5px; }
.legend { display:flex; flex-wrap:wrap; gap:.6rem 1.4rem; margin:.8rem 0;
  font-size:.86rem; }
.legend label { display:flex; align-items:center; gap:.45rem; cursor:pointer; }
.legend input { accent-color:var(--accent); }
.swatch { width:.85rem; height:.85rem; border-radius:3px; display:inline-block; }
.stats { font-size:.76rem; color:var(--muted); margin-top:.4rem;
  font-family:ui-monospace,Menlo,monospace; }
input[type=checkbox]:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
</style></head><body>
<main>
<h1>__TITLE__</h1>
<div id="stage">
  <canvas id="gl"></canvas>
  <div class="hint">ドラッグ: 回転 · ホイール: ズーム</div>
  <div class="axis-tag">z軸 = ドライバ → ピンナ</div>
</div>
<div class="legend">__LEGEND__
  <label><span class="swatch" style="background:#54C6D6;border-radius:50%"></span>\
受音プローブ</label>
  <label><span class="swatch" style="background:#5B8DD9;border-radius:50%"></span>\
ドライバ振動面</label>
</div>
<div class="stats">__STATS__</div>
</main>
<script>
"use strict";
const PARTS = __PARTS__;
const META = __META__;
const PROBES_B64 = "__PROBES__";
function decodeB64(b64, Type) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}
const hexToRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
const dx = META.dxMm;
const center = META.extentMm.map(e => e / 2);
center[2] = META.driverCenter[2] * 0.35 + center[2] * 0.65;
function buildPart(faceData) {
  const f = decodeB64(faceData, Uint16Array);
  const n = f.length / 4;
  const pos = new Float32Array(n * 18), nrm = new Float32Array(n * 18);
  const AX = [[1, 2], [0, 2], [0, 1]];
  const v = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]];
  let o = 0;
  for (let i = 0; i < n; i++) {
    const axisSign = f[i*4], plane = f[i*4+1], a = f[i*4+2], b = f[i*4+3];
    const axis = axisSign >> 1, sign = (axisSign & 1) ? -1 : 1;
    const ua = AX[axis][0], ub = AX[axis][1];
    for (let c = 0; c < 4; c++) {
      const da = (c === 1 || c === 2) ? 1 : 0, db = (c >= 2) ? 1 : 0;
      v[c][axis] = plane * dx; v[c][ua] = (a + da) * dx; v[c][ub] = (b + db) * dx;
    }
    const tri = [0, 1, 2, 0, 2, 3];
    for (let t = 0; t < 6; t++) {
      const p = v[tri[t]];
      pos[o] = p[0]-center[0]; pos[o+1] = p[1]-center[1]; pos[o+2] = p[2]-center[2];
      nrm[o] = 0; nrm[o+1] = 0; nrm[o+2] = 0; nrm[o + axis] = sign;
      o += 3;
    }
  }
  return { pos, nrm, count: n * 6 };
}
function persp(fov, asp, near, far) {
  const t = 1 / Math.tan(fov / 2), d = 1 / (near - far);
  return [t/asp,0,0,0, 0,t,0,0, 0,0,(near+far)*d,-1, 0,0,2*near*far*d,0];
}
const sub3=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const dot3=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm3=a=>{const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];};
function lookAt(eye, at, up) {
  const z = norm3(sub3(eye, at)), x = norm3(cross3(up, z)), y = cross3(z, x);
  return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
          -dot3(x,eye),-dot3(y,eye),-dot3(z,eye),1];
}
function mul4(a, b) {
  const r = new Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k*4+j] * b[i*4+k];
    r[i*4+j] = s;
  }
  return r;
}
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl", { antialias: true });
const VS = `attribute vec3 aPos; attribute vec3 aNrm;
uniform mat4 uMvp; uniform float uPointSize; varying vec3 vNrm;
void main(){ gl_Position = uMvp*vec4(aPos,1.0); vNrm = aNrm; gl_PointSize = uPointSize; }`;
const FS = `precision mediump float;
uniform vec3 uColor; uniform float uAlpha; uniform float uLit; varying vec3 vNrm;
void main(){
  vec3 L1 = normalize(vec3(0.5, 0.7, -0.6));
  vec3 L2 = normalize(vec3(-0.6, -0.2, 0.75));
  float lit = 0.34 + 0.48*abs(dot(vNrm, L1)) + 0.30*abs(dot(vNrm, L2));
  gl_FragColor = vec4(mix(uColor, uColor*lit, uLit), uAlpha);
}`;
function shader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s);
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog); gl.useProgram(prog);
const loc = {
  pos: gl.getAttribLocation(prog, "aPos"), nrm: gl.getAttribLocation(prog, "aNrm"),
  mvp: gl.getUniformLocation(prog, "uMvp"), color: gl.getUniformLocation(prog, "uColor"),
  alpha: gl.getUniformLocation(prog, "uAlpha"), lit: gl.getUniformLocation(prog, "uLit"),
  psize: gl.getUniformLocation(prog, "uPointSize"),
};
function makeBuffers(pos, nrm) {
  const pb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
  const nb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, nb); gl.bufferData(gl.ARRAY_BUFFER, nrm, gl.STATIC_DRAW);
  return { pb, nb };
}
const drawables = [];
for (const part of PARTS) {
  const { pos, nrm, count } = buildPart(part.data);
  drawables.push({ name: part.name, ...makeBuffers(pos, nrm), count,
                   color: hexToRgb(part.color), visible: true, mode: "tris" });
}
{
  const probesArr = decodeB64(PROBES_B64, Float32Array);
  const probePos = new Float32Array(probesArr.length);
  for (let i = 0; i < probesArr.length; i += 3) {
    probePos[i] = probesArr[i]-center[0];
    probePos[i+1] = probesArr[i+1]-center[1];
    probePos[i+2] = probesArr[i+2]-center[2];
  }
  drawables.push({ name: "probes",
    ...makeBuffers(probePos, new Float32Array(probePos.length)),
    count: probePos.length / 3, color: hexToRgb("#54C6D6"), visible: true, mode: "points" });
}
{
  const seg = 64, r = META.driverRadiusMm, c = META.driverCenter;
  const pos = new Float32Array((seg + 2) * 3), nrm = new Float32Array((seg + 2) * 3);
  pos[0] = c[0]-center[0]; pos[1] = c[1]-center[1]; pos[2] = c[2]-center[2];
  for (let i = 0; i <= seg; i++) {
    const th = i / seg * Math.PI * 2, o = (i + 1) * 3;
    pos[o] = c[0] + r*Math.cos(th) - center[0];
    pos[o+1] = c[1] + r*Math.sin(th) - center[1];
    pos[o+2] = c[2] - center[2];
  }
  for (let i = 0; i < nrm.length; i += 3) nrm[i+2] = 1;
  drawables.push({ name: "driver", ...makeBuffers(pos, nrm), count: seg + 2,
    color: hexToRgb("#5B8DD9"), visible: true, mode: "fan" });
}
let az = 1.9, el = 0.35, dist = 220;
function render() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth * dpr, h = canvas.clientHeight * dpr;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  gl.clearColor(0.039, 0.059, 0.075, 1);
  gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  const ce = Math.cos(el);
  const eye = [dist*Math.sin(az)*ce, dist*Math.sin(el), dist*Math.cos(az)*ce];
  const mvp = mul4(persp(0.75, w / h, 1, 3000), lookAt(eye, [0, 0, 0], [0, 1, 0]));
  gl.uniformMatrix4fv(loc.mvp, false, new Float32Array(mvp));
  for (const d of drawables) {
    if (!d.visible) continue;
    gl.bindBuffer(gl.ARRAY_BUFFER, d.pb);
    gl.enableVertexAttribArray(loc.pos);
    gl.vertexAttribPointer(loc.pos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, d.nb);
    gl.enableVertexAttribArray(loc.nrm);
    gl.vertexAttribPointer(loc.nrm, 3, gl.FLOAT, false, 0, 0);
    gl.uniform3fv(loc.color, d.color);
    if (d.mode === "points") {
      gl.uniform1f(loc.lit, 0); gl.uniform1f(loc.alpha, 1); gl.uniform1f(loc.psize, 4*dpr);
      gl.drawArrays(gl.POINTS, 0, d.count);
    } else if (d.mode === "fan") {
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.uniform1f(loc.lit, 0.35); gl.uniform1f(loc.alpha, 0.42); gl.uniform1f(loc.psize, 1);
      gl.drawArrays(gl.TRIANGLE_FAN, 0, d.count);
      gl.depthMask(true); gl.disable(gl.BLEND);
    } else {
      gl.uniform1f(loc.lit, 1); gl.uniform1f(loc.alpha, 1); gl.uniform1f(loc.psize, 1);
      gl.drawArrays(gl.TRIANGLES, 0, d.count);
    }
  }
}
let dragging = false, px = 0, py = 0;
canvas.addEventListener("pointerdown", e => { dragging = true; px = e.clientX;
  py = e.clientY; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", e => {
  if (!dragging) return;
  az -= (e.clientX - px) * 0.008; el += (e.clientY - py) * 0.008;
  el = Math.max(-1.5, Math.min(1.5, el)); px = e.clientX; py = e.clientY;
  requestAnimationFrame(render);
});
canvas.addEventListener("pointerup", () => { dragging = false; });
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  dist = Math.max(40, Math.min(900, dist * Math.exp(e.deltaY * 0.0012)));
  requestAnimationFrame(render);
}, { passive: false });
document.querySelectorAll("input[data-part]").forEach(cb => {
  cb.addEventListener("change", () => {
    const d = drawables.find(x => x.name === cb.dataset.part);
    if (d) d.visible = cb.checked;
    requestAnimationFrame(render);
  });
});
new ResizeObserver(() => requestAnimationFrame(render)).observe(canvas);
render();
</script></body></html>
"""
