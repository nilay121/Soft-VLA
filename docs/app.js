// Soft Arm Playground: UI, rendering and the real-time loop around sim.js.

import loadMujoco from './vendor/mujoco.js';
import { SoftArmSim, loadScene, balancedPressure } from './sim.js';

const THREE = window.THREE;

// Regulator slew limits, as in VLA_control/continual_pick_place/combined_deploy.py
const ARM_RATE = 1.2;     // bar/s
const GRIP_RATE = 2.0;    // bar/s
const FRAME_BUDGET_MS = 14;
const START_LEVEL = 0.4;  // bar, the balanced start pose every experiment uses

const $ = id => document.getElementById(id);
const loadingMsg = msg => { $('loading-msg').textContent = msg; };

function fail(err) {
  console.error(err);
  const box = $('loading');
  box.hidden = false;
  box.classList.add('error');
  box.querySelector('b').textContent = 'The simulator could not start';
  loadingMsg(`${err && err.message ? err.message : err}. Reload the page to try again.`);
}

async function fetchBytes(path) {
  const r = await fetch(`assets/${path}`);
  if (!r.ok) throw new Error(`could not load assets/${path} (HTTP ${r.status})`);
  return new Uint8Array(await r.arrayBuffer());
}
const fetchJSON = async path => JSON.parse(new TextDecoder().decode(await fetchBytes(path)));

const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// -------------------------------------------------------------------------
// boot
// -------------------------------------------------------------------------

async function boot() {
  loadingMsg('MuJoCo physics engine, 10 MB');
  const mj = await loadMujoco({ locateFile: f => new URL(`vendor/${f}`, document.baseURI).href });
  loadingMsg('Robot, gripper and scene');
  const [C, layouts, expert] = await Promise.all(
    [fetchJSON('constants.json'), fetchJSON('layouts.json'), fetchJSON('expert.json')]);
  const model = await loadScene(mj, fetchBytes);
  const sim = new SoftArmSim(mj, model, C);
  const app = new App(mj, sim, C, layouts, expert);
  loadingMsg('Letting the arm settle');
  await new Promise(r => setTimeout(r, 30));
  app.newEpisode(0);
  $('loading').hidden = true;
  app.start();
  window.softArm = app;     // for the browser console

}

// -------------------------------------------------------------------------
// the app
// -------------------------------------------------------------------------

class App {
  constructor(mj, sim, C, layouts, expert) {
    Object.assign(this, { mj, sim, C, layouts, expert });
    this.target = new Float64Array(7);    // what the sliders ask for
    this.applied = new Float64Array(7);   // what the chambers hold now
    this.replay = null;                   // {object, commands, k}
    this.layoutIndex = 0;
    this.speedWindow = { sim: 0, wall: 0, shown: 1 };
    this.buildControls();
    this.view = new View(mj, sim, C, $('view'), $('stage'));
    this.view.app = this;     // the chamber glow reads the live pressures
  }

  // --- controls ----------------------------------------------------------

  buildControls() {
    const chamberColour = ['--ch1', '--ch2', '--ch3'];
    this.sliders = [];
    const make = (parent, i, label, max, colour) => {
      const row = document.createElement('div');
      row.className = 'reg';
      const id = `p-${label}`;
      row.innerHTML = `<label for="${id}">${label}</label>
        <input type="range" id="${id}" min="0" max="${max}" step="0.01" value="0" style="--c: var(${colour})">
        <output for="${id}"><b>0.00</b> → 0.00</output>`;
      parent.appendChild(row);
      const input = row.querySelector('input');
      input.addEventListener('input', () => {
        if (this.replay) this.stopReplay();
        this.target[i] = parseFloat(input.value);
      });
      this.sliders[i] = { input, out: row.querySelector('output') };
    };
    for (let m = 0; m < 2; m++) {
      for (let c = 0; c < 3; c++) {
        make($(`sliders-${m}`), 3 * m + c, `${'AB'[m]}${c + 1}`, this.C.arm.p_max, chamberColour[c]);
      }
    }
    make($('sliders-g'), 6, 'G', this.C.gripper.p_max, '--grip');

    $('btn-straight').addEventListener('click', () => {
      this.stopReplay();
      balancedPressure(this.C, START_LEVEL).forEach((p, i) => { this.target[i] = p; });
      this.syncSliders();
    });
    $('btn-reset').addEventListener('click', () => this.newEpisode(this.layoutIndex));
    $('btn-layout').addEventListener('click', () => {
      const n = this.layouts.length;
      this.newEpisode(1 + Math.floor(Math.random() * (n - 1)));
    });
    for (const name of Object.keys(this.expert.episodes)) {
      const b = document.createElement('button');
      b.textContent = name;
      b.addEventListener('click', () => this.startReplay(name));
      $('expert-buttons').appendChild(b);
    }
    $('wrist-toggle').addEventListener('change', e => { this.view.showWrist = e.target.checked; });
    this.drawCrossSections(true);
  }

  syncSliders() {
    this.sliders.forEach((s, i) => { s.input.value = this.target[i].toFixed(2); });
  }

  // --- episodes ----------------------------------------------------------

  newEpisode(index, armStart = null) {
    this.stopReplay();
    this.layoutIndex = index;
    const L = this.layouts[index];
    this.sim.setLayout(L.objects, L.bowl_xy);
    const start = armStart || balancedPressure(this.C, START_LEVEL);
    this.sim.reset(start, 0);
    start.forEach((p, i) => { this.target[i] = p; this.applied[i] = p; });
    this.target[6] = this.applied[6] = 0;
    this.syncSliders();
    this.restZ = {};
    for (const n of this.C.task_objects) this.restZ[n] = this.sim.objectPos(n)[2];
  }

  startReplay(name) {
    const ep = this.expert.episodes[name];
    const lay = this.expert.layout;
    // the expert's layout is layouts[0]; reset exactly as it was recorded
    const index = this.layouts.findIndex(l =>
      JSON.stringify(l.objects) === JSON.stringify(lay.objects));
    this.newEpisode(index >= 0 ? index : 0, balancedPressure(this.C, ep.p0));
    this.replay = { object: name, commands: ep.commands, k: 0 };
    $('expert-note').textContent = `Expert is fetching the ${name}. Touch any slider to take over.`;
  }

  stopReplay(done = false) {
    if (!this.replay) return;
    const name = this.replay.object;
    this.replay = null;
    $('expert-progress').style.width = '0';
    $('expert-note').textContent = done
      ? `The expert finished the ${name}. The sliders are yours again.`
      : 'You have control.';
  }

  // --- one 10 ms control step -------------------------------------------

  controlStep(dt) {
    if (this.replay) {
      const r = this.replay;
      const c = r.commands[r.k++];
      for (let i = 0; i < 7; i++) { this.applied[i] = c[i]; this.target[i] = c[i]; }
      if (r.k >= r.commands.length) this.stopReplay(true);
    } else {
      for (let i = 0; i < 7; i++) {
        const rate = (i === 6 ? GRIP_RATE : ARM_RATE) * dt;
        const d = this.target[i] - this.applied[i];
        this.applied[i] += Math.max(-rate, Math.min(rate, d));
      }
    }
    this.sim.apply(this.applied.subarray(0, 6), this.applied[6]);
    this.sim.step();
  }

  // --- loop --------------------------------------------------------------

  start() {
    const dt = this.sim.C.scene.control_dt;
    let last = performance.now(), owed = 0;
    const frame = now => {
      const wall = Math.min((now - last) / 1000, 0.1);
      last = now;
      owed += wall;
      // On a slow GPU frames get long; let the physics use most of each
      // frame then, so simulated time still keeps up with the clock.
      const budget = Math.min(60, Math.max(FRAME_BUDGET_MS, 0.7 * wall * 1000));
      const until = performance.now() + budget;
      let steps = 0;
      while (owed >= dt && performance.now() < until) {
        this.controlStep(dt);
        owed -= dt;
        steps++;
      }
      owed = Math.min(owed, 5 * dt);          // fall behind gracefully
      this.trackSpeed(steps * dt, wall);
      this.view.sync();
      this.view.render();
      this.updateReadouts();
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  trackSpeed(simS, wallS) {
    const w = this.speedWindow;
    w.sim += simS; w.wall += wallS;
    if (w.wall > 0.5) {
      w.shown = w.sim / w.wall;
      w.sim = w.wall = 0;
    }
  }

  updateReadouts() {
    this.sliders.forEach((s, i) => {
      const live = this.applied[i].toFixed(2), want = this.target[i].toFixed(2);
      const text = `<b>${live}</b> → ${want}`;
      if (s.last !== text) { s.out.innerHTML = text; s.last = text; }
      if (this.replay) s.input.value = want;
    });
    if (this.replay) {
      $('expert-progress').style.width = `${(100 * this.replay.k / this.replay.commands.length).toFixed(1)}%`;
    }
    const tcp = this.sim.tcp();
    const mm = v => `${Math.round(v * 1000) || 0}`;      // no "-0"
    $('hud-tcp').textContent =
      `gripper ${mm(tcp[2] - this.C.scene.table_top)} mm above table · x ${mm(tcp[0])} y ${mm(tcp[1])}`;
    const speed = this.speedWindow.shown;
    $('hud-speed').textContent = speed > 0.97 ? 'real time' : `${speed.toFixed(2)}× real time`;
    this.updateChips();
    this.drawCrossSections();
  }

  updateChips() {
    const parts = this.C.task_objects.map(n => {
      const state = this.sim.inBowl(n) ? 'in bowl'
        : this.sim.objectPos(n)[2] > this.restZ[n] + 0.008 ? 'lifted' : 'on table';
      return { n, state };
    });
    const key = parts.map(p => p.n + p.state).join();
    if (key === this.chipKey) return;
    this.chipKey = key;
    $('chips').innerHTML = parts.map(p =>
      `<span class="chip" data-state="${p.state}">${p.n}: ${p.state}</span>`).join('');
  }

  // --- chamber cross-sections -------------------------------------------
  // The three chambers at their THETA angles, filled by pressure, and an
  // arrow for the direction the module bends (away from the stronger side).

  drawCrossSections(force = false) {
    const K = this.C.kinematics;
    for (let m = 0; m < 2; m++) {
      const p = [0, 1, 2].map(i => this.applied[3 * m + i]);
      const key = p.map(v => v.toFixed(2)).join();
      const svg = $(`xsec-${m}`);
      if (!force && svg.dataset.key === key) continue;
      svg.dataset.key = key;
      const cols = ['--ch1', '--ch2', '--ch3'].map(cssVar);
      let bx = 0, by = 0, parts = '';
      parts += `<circle r="30" fill="none" stroke="${cssVar('--line')}" stroke-width="2"/>`;
      for (let i = 0; i < 3; i++) {
        const th = K.theta_rad[i];
        // Seen from above. The arm hangs (base turned 180 deg about x), so its
        // frame's +y is world -y, which is screen-down: arm coordinates map
        // straight onto the screen.
        const x = 17 * Math.cos(th), y = 17 * Math.sin(th);
        const eff = p[i] >= K.p_threshold ? p[i] * K.strain_gain[i] : 0;
        bx -= eff * Math.cos(th); by -= eff * Math.sin(th);
        const fill = Math.min(1, p[i] / this.C.arm.p_max);
        parts += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="8" fill="${cols[i]}" fill-opacity="${(0.12 + 0.88 * fill).toFixed(2)}" stroke="${cols[i]}" stroke-width="1.5"/>`;
        parts += `<text x="${(x * 1.85).toFixed(1)}" y="${(y * 1.85 + 3).toFixed(1)}" text-anchor="middle">${'AB'[m]}${i + 1}</text>`;
      }
      const mag = Math.hypot(bx, by);
      if (mag > 0.05) {
        const len = Math.min(13, 4 + 6 * mag);
        const ux = bx / mag, uy = by / mag;
        const ex = ux * len, ey = uy * len;
        parts += `<line x1="0" y1="0" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}" stroke="${cssVar('--ink')}" stroke-width="2" stroke-linecap="round"/>`;
        const hx = -uy * 3, hy = ux * 3;
        parts += `<path d="M${(ex + ux * 4).toFixed(1)},${(ey + uy * 4).toFixed(1)} L${(ex + hx).toFixed(1)},${(ey + hy).toFixed(1)} L${(ex - hx).toFixed(1)},${(ey - hy).toFixed(1)} Z" fill="${cssVar('--ink')}"/>`;
      } else {
        parts += `<circle r="2.5" fill="${cssVar('--ink')}"/>`;
      }
      svg.innerHTML = parts;
    }
  }
}

// -------------------------------------------------------------------------
// three.js view of the MuJoCo scene
// -------------------------------------------------------------------------

class View {
  constructor(mj, sim, C, canvas, stage) {
    Object.assign(this, { mj, sim, C, canvas, stage });
    this.showWrist = true;
    const model = sim.model;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputEncoding = THREE.sRGBEncoding;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 4 / 3, 0.01, 20);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.78, -0.98, 0.86);
    this.controls = new THREE.OrbitControls(this.camera, canvas);
    this.controls.target.set(0.0, 0.0, 0.53);
    this.controls.enableDamping = true;
    this.controls.minDistance = 0.25;
    this.controls.maxDistance = 3;
    this.controls.update();

    const wristId = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA.value, 'wrist_camera');
    this.wristId = wristId;
    this.wristCam = new THREE.PerspectiveCamera(wristId >= 0 ? model.cam_fovy[wristId] : 60, 1, 0.005, 5);

    this.scene.add(new THREE.HemisphereLight(0xf2f5f8, 0x5a5048, 0.75));
    const sun = new THREE.DirectionalLight(0xffffff, 0.85);
    sun.position.set(0.5, -0.6, 1.8);
    sun.target.position.set(0, 0, 0.37);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -0.45, right: 0.45, top: 0.45, bottom: -0.45, near: 0.5, far: 3 });
    sun.shadow.bias = -0.0005;
    this.scene.add(sun, sun.target);

    this.buildGeoms();
    this.applyTheme();
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.applyTheme());
    new MutationObserver(() => this.applyTheme())
      .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    new ResizeObserver(() => this.resize()).observe(stage);
    this.resize();
  }

  applyTheme() {
    this.scene.background = new THREE.Color(cssVar('--viewport') || '#cfd6de');
    const cols = ['--ch1', '--ch2', '--ch3'].map(v => new THREE.Color(cssVar(v)));
    for (const c of this.cavities) c.mesh.material.color.copy(cols[c.chamber % 3]);
  }

  buildGeoms() {
    const { mj, sim } = this;
    const model = sim.model;
    const T = mj.mjtGeom;
    const name = i => mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM.value, i) || '';
    const meshName = i => mj.mj_id2name(model, mj.mjtObj.mjOBJ_MESH.value, i) || '';
    const bodyName = i => mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, i) || '';
    const rgbaAll = model.geom_rgba, matRgba = model.mat_rgba;
    const sizeAll = model.geom_size;
    const vert = model.mesh_vert, face = model.mesh_face;
    const cavityForChamber = this.C.arm.cavity_for_chamber;   // chamber i -> cavity mesh number

    this.items = [];
    this.cavities = [];
    for (let g = 0; g < model.ngeom; g++) {
      const n = name(g);
      if (n.startsWith('link_')) continue;                    // mass carriers, like hide_link_capsules
      const mat = model.geom_matid[g];
      let rgba = Array.from(rgbaAll.subarray(4 * g, 4 * g + 4));
      if (mat >= 0) rgba = Array.from(matRgba.subarray(4 * mat, 4 * mat + 4));
      if (rgba[3] <= 0.01) continue;
      const type = model.geom_type[g];
      const s = Array.from(sizeAll.subarray(3 * g, 3 * g + 3));
      let geo;
      if (type === T.mjGEOM_PLANE.value) {
        geo = new THREE.PlaneGeometry(2 * (s[0] || 2), 2 * (s[1] || 2));
      } else if (type === T.mjGEOM_SPHERE.value) {
        geo = new THREE.SphereGeometry(s[0], 24, 16);
      } else if (type === T.mjGEOM_ELLIPSOID.value) {
        geo = new THREE.SphereGeometry(1, 24, 16);
        geo.scale(s[0], s[1], s[2]);
      } else if (type === T.mjGEOM_CAPSULE.value) {
        geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 16);
        geo.rotateX(Math.PI / 2);
      } else if (type === T.mjGEOM_CYLINDER.value) {
        geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 28);
        geo.rotateX(Math.PI / 2);
      } else if (type === T.mjGEOM_BOX.value) {
        geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
      } else if (type === T.mjGEOM_MESH.value) {
        const id = model.geom_dataid[g];
        const va = model.mesh_vertadr[id], fa = model.mesh_faceadr[id], fn = model.mesh_facenum[id];
        const pos = new Float32Array(9 * fn);
        for (let f = 0; f < fn; f++) {
          for (let k = 0; k < 3; k++) {
            const v = va + face[3 * (fa + f) + k];
            pos[9 * f + 3 * k] = vert[3 * v];
            pos[9 * f + 3 * k + 1] = vert[3 * v + 1];
            pos[9 * f + 3 * k + 2] = vert[3 * v + 2];
          }
        }
        geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.computeVertexNormals();
      } else {
        continue;
      }

      const colour = new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2]);
      let opacity = rgba[3];
      const material = new THREE.MeshStandardMaterial({
        color: colour, roughness: 0.65, metalness: 0.02,
        flatShading: type === T.mjGEOM_MESH.value,
      });

      // The I-Support shell is drawn translucent so the air chambers inside
      // it show; the chambers are tinted by the pressure in them.
      let cavity = null;
      if (type === T.mjGEOM_MESH.value) {
        const mn = meshName(model.geom_dataid[g]);
        const cav = mn.match(/cavity(\d)/);
        if (cav) {
          const body = bodyName(model.geom_bodyid[g]);
          const seg = body === 'robot_base' ? 0 : parseInt(body.split('_')[1], 10);
          const module = seg > this.C.arm.segments_per_module ? 1 : 0;
          const chamber = cavityForChamber.indexOf(parseInt(cav[1], 10));
          cavity = { chamber: 3 * module + chamber };
          opacity = 0.9;
          material.emissive = new THREE.Color(0, 0, 0);
        } else if (/^(border|chambers|fixed_border|fixed_chambers)/.test(mn)) {
          material.color.setRGB(0.70, 0.75, 0.82);
          opacity = 0.34;
          material.depthWrite = false;
        }
      }
      if (opacity < 0.999) { material.transparent = true; material.opacity = opacity; }

      const mesh = new THREE.Mesh(geo, material);
      mesh.matrixAutoUpdate = false;
      const solid = opacity > 0.6;
      mesh.castShadow = solid && type !== T.mjGEOM_PLANE.value;
      mesh.receiveShadow = true;
      if (cavity) { cavity.mesh = mesh; this.cavities.push(cavity); }
      this.items.push({ g, mesh });
      this.scene.add(mesh);
    }
  }

  // copy MuJoCo's geom poses into three.js
  sync() {
    const data = this.sim.data;
    const xpos = data.geom_xpos, xmat = data.geom_xmat;
    const m4 = new THREE.Matrix4();
    for (const { g, mesh } of this.items) {
      const R = xmat.subarray(9 * g, 9 * g + 9), p = xpos.subarray(3 * g, 3 * g + 3);
      m4.set(R[0], R[1], R[2], p[0],
             R[3], R[4], R[5], p[1],
             R[6], R[7], R[8], p[2],
             0, 0, 0, 1);
      mesh.matrix.copy(m4);
      mesh.matrixWorldNeedsUpdate = true;
    }
    // chamber glow follows its pressure (applied by the App each step)
    const app = this.app;
    if (app) {
      for (const c of this.cavities) {
        const p = app.applied[c.chamber] / this.C.arm.p_max;
        c.mesh.material.emissive.copy(c.mesh.material.color).multiplyScalar(0.55 * p);
      }
    }
    if (this.wristId >= 0) {
      const cp = data.cam_xpos.subarray(3 * this.wristId, 3 * this.wristId + 3);
      const R = data.cam_xmat.subarray(9 * this.wristId, 9 * this.wristId + 9);
      // MuJoCo cameras look down their -z with +y up, as three.js cameras do
      m4.set(R[0], R[1], R[2], cp[0], R[3], R[4], R[5], cp[1], R[6], R[7], R[8], cp[2], 0, 0, 0, 1);
      this.wristCam.matrixAutoUpdate = false;
      this.wristCam.matrix.copy(m4);
      this.wristCam.matrixWorldNeedsUpdate = true;
      this.wristCam.updateMatrixWorld(true);
    }
  }

  resize() {
    const w = this.stage.clientWidth, h = this.stage.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.size = { w, h };
  }

  render() {
    if (!this.size) return;
    const { w, h } = this.size;
    const r = this.renderer;
    this.controls.update();
    r.setScissorTest(false);
    r.setViewport(0, 0, w, h);
    r.render(this.scene, this.camera);
    const label = $('wrist-label');
    if (this.showWrist && this.wristId >= 0) {
      const s = Math.round(Math.min(w, h) * 0.3), pad = 12;
      r.setScissorTest(true);
      r.setScissor(w - s - pad, pad, s, s);
      r.setViewport(w - s - pad, pad, s, s);
      r.render(this.scene, this.wristCam);
      r.setScissorTest(false);
      label.hidden = false;
      label.style.bottom = `${pad + s + 4}px`;
      label.style.right = `${pad}px`;
    } else {
      label.hidden = true;
    }
  }
}

boot().catch(fail);
