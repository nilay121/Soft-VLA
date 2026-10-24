// The two-module I-Support simulator in the browser.
//
// A line-by-line port of the Python side, driven by MuJoCo's own WASM build
// on the unchanged scene_continual.xml:
//
//   pressureToTargets   sim/kinematics.py + robot.TwoModulePressureMapper
//   fingerTargets       gripper.finger_targets
//   stabilize           sim/mujoco_env.stabilize_model
//   setLayout / reset   continual_scene.ContinualScene.set_multi_layout / reset
//   inBowl              continual_scene.ContinualScene.in_bowl
//
// Every number comes from assets/constants.json, which build_web_demo.py
// reads out of the simulator modules.

export class SoftArmSim {
  constructor(mj, model, C) {
    this.mj = mj;
    this.model = model;
    this.data = new mj.MjData(model);
    this.C = C;

    const id = (type, name) => mj.mj_name2id(model, mj.mjtObj[type].value, name);
    this.tcpSite = id('mjOBJ_SITE', 'tcp');
    this.bowlBody = id('mjOBJ_BODY', 'bowl');
    this.gripperActuators = C.gripper.joint_names.map(n => id('mjOBJ_ACTUATOR', `act_${n}`));
    if (this.tcpSite < 0 || this.bowlBody < 0 || this.gripperActuators.some(a => a < 0)) {
      throw new Error('scene is missing the tcp site, the bowl or a gripper actuator');
    }
    this.objects = {};
    for (const [name, spec] of Object.entries(C.objects)) {
      const body = id('mjOBJ_BODY', `obj_${name}`);
      if (body < 0) throw new Error(`scene has no body obj_${name}`);
      const qadr = model.jnt_qposadr[model.body_jntadr[body]];
      this.objects[name] = { ...spec, body, qadr };
    }
    this.substeps = Math.max(1, Math.round(C.scene.control_dt / model.opt.timestep));
    this.ctrl = new Float64Array(model.nu);
    this.stabilize();
  }

  // --- sim/mujoco_env.stabilize_model ------------------------------------
  stabilize() {
    const { mj, model } = this;
    const S = this.C.stabilize;
    const damping = model.dof_damping, armature = model.dof_armature;
    for (let i = 0; i < model.nv; i++) {
      damping[i] = Math.max(damping[i] * S.joint_damping_scale, S.min_joint_damping);
      armature[i] = Math.max(armature[i], S.dof_armature);
    }
    const affine = mj.mjtBias.mjBIAS_AFFINE.value;
    for (let a = 0; a < model.nu; a++) {
      const joint = model.actuator_trnid[2 * a];
      if (joint < 0 || joint >= model.njnt) continue;
      if (model.actuator_biastype[a] !== affine) continue;
      const kp = model.actuator_gainprm[10 * a];
      if (kp <= 0) continue;
      const inertia = Math.max(model.dof_M0[model.jnt_dofadr[joint]], 1e-8);
      model.actuator_biasprm[10 * a + 2] = -2 * S.actuator_damping_ratio * Math.sqrt(kp * inertia);
    }
    model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST.value;
    model.opt.tolerance = S.tolerance;
    model.opt.iterations = S.iterations;
    model.opt.ls_iterations = S.ls_iterations;
  }

  // --- sim/kinematics.py, once per module ---------------------------------
  // Returns [extension, kappa_x, kappa_y] for one module's three pressures.
  moduleDeformation(p3) {
    const K = this.C.kinematics;
    const coeff = K.air_area / (K.youngs_modulus * K.actuator_area);
    const strain = p3.map((p, i) =>
      K.strain_gain[i] * coeff * (p >= K.p_threshold ? p : 0) * K.bar_to_pa);
    const mean = (strain[0] + strain[1] + strain[2]) / 3;
    let kx = 0, ky = 0;
    for (let i = 0; i < 3; i++) {
      kx += strain[i] * K.actuator_area * K.delta * Math.sin(K.theta_rad[i]);
      ky += strain[i] * K.actuator_area * K.delta * Math.cos(K.theta_rad[i]);
    }
    return [mean * K.total_length, kx / K.I1, -ky / K.I2];
  }

  // robot.TwoModulePressureMapper.convert: 6 pressures -> 30 setpoints,
  // ordered per segment (bend_x, bend_y, extension).
  pressureToTargets(arm, out) {
    const K = this.C.kinematics, A = this.C.arm;
    for (let m = 0; m < A.modules; m++) {
      const p3 = [0, 1, 2].map(i => Math.min(A.p_max, Math.max(A.p_min, arm[3 * m + i])));
      const [ext, kx, ky] = this.moduleDeformation(p3);
      for (let s = 0; s < A.segments_per_module; s++) {
        const base = 3 * (m * A.segments_per_module + s);
        out[base] = kx * K.segment_length;
        out[base + 1] = ky * K.segment_length;
        out[base + 2] = ext / K.num_segments;
      }
    }
  }

  // gripper.finger_targets: one line pressure -> 12 hinge angles [rad].
  fingerTargets(p) {
    const G = this.C.gripper;
    const x = Math.min(Math.max(p, 0), G.p_max) / G.p_max;
    const perJoint = new Array(G.segments).fill(-G.close_per_joint_deg * Math.PI / 180 * x);
    perJoint[0] += G.open_splay_deg * Math.PI / 180 * (1 - x);
    const out = [];
    for (let f = 0; f < G.fingers; f++) out.push(...perJoint);
    return out;
  }

  apply(arm, grip) {
    this.ctrl.fill(0);
    this.pressureToTargets(arm, this.ctrl);
    const fingers = this.fingerTargets(grip);
    this.gripperActuators.forEach((a, i) => { this.ctrl[a] = fingers[i]; });
    this.data.ctrl.set(this.ctrl);
  }

  step() {
    for (let i = 0; i < this.substeps; i++) this.mj.mj_step(this.model, this.data);
  }

  // continual_scene.ContinualScene.set_multi_layout
  setLayout(objectsXY, bowlXY) {
    const { model } = this;
    const top = this.C.scene.table_top;
    for (const [name, o] of Object.entries(this.objects)) {
      const xy = objectsXY[name];
      const pos = xy ? [xy[0], xy[1], top + o.rest + 1e-4]
                     : [o.parking[0], o.parking[1], o.rest + 1e-4];
      model.qpos0.set([...pos, 1, 0, 0, 0], o.qadr);
    }
    if (bowlXY) {
      model.body_pos[3 * this.bowlBody] = bowlXY[0];
      model.body_pos[3 * this.bowlBody + 1] = bowlXY[1];
    }
  }

  // hanging_robot.Scene.reset
  reset(arm, grip = 0) {
    const { mj, model, data } = this;
    mj.mj_resetData(model, data);
    this.apply(arm, grip);
    mj.mj_forward(model, data);
    const n = Math.round(this.C.scene.settle_s / model.opt.timestep);
    for (let i = 0; i < n; i++) mj.mj_step(model, data);
  }

  objectPos(name) {
    const b = this.objects[name].body;
    return Array.from(this.data.xipos.subarray(3 * b, 3 * b + 3));
  }

  bowlCentre() {
    return Array.from(this.data.xpos.subarray(3 * this.bowlBody, 3 * this.bowlBody + 3));
  }

  tcp() {
    return Array.from(this.data.site_xpos.subarray(3 * this.tcpSite, 3 * this.tcpSite + 3));
  }

  // continual_scene.ContinualScene.in_bowl
  inBowl(name) {
    const c = this.bowlCentre(), p = this.objectPos(name);
    const radial = Math.hypot(p[0] - c[0], p[1] - c[1]);
    return radial < this.C.scene.bowl_r_rim && p[2] < c[2] - this.C.scene.bowl_in_margin;
  }
}

// hanging_robot.balanced_pressure: both modules straight down, each chamber
// scaled by 1/strain-gain so the three bend terms cancel.
export function balancedPressure(C, level) {
  const g = C.kinematics.strain_gain, gmin = Math.min(...g);
  const one = g.map(x => level * gmin / x);
  return [...one, ...one];
}

// Load scene.xml and its meshes into MuJoCo's virtual file system.
// `fetchBytes(path)` returns a Uint8Array for a path under assets/. The
// meshes come as one JSON bundle {file: base64} (see build_web_demo.py).
export async function loadScene(mj, fetchBytes) {
  const [xmlBytes, bundleBytes] = await Promise.all(
    [fetchBytes('scene.xml'), fetchBytes('meshes.json')]);
  const xml = new TextDecoder().decode(xmlBytes);
  const bundle = JSON.parse(new TextDecoder().decode(bundleBytes));
  const meshdir = (xml.match(/meshdir="([^"]+)"/) || [, ''])[1];
  const files = [...new Set([...xml.matchAll(/file="([^"]+)"/g)].map(m => m[1]))];
  const vfs = new mj.MjVFS();
  for (const f of files) {
    if (!(f in bundle)) throw new Error(`mesh ${f} missing from meshes.json`);
    const bin = atob(bundle[f]);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    vfs.addBuffer(`${meshdir}/${f}`, bytes);
  }
  vfs.addBuffer('scene.xml', xmlBytes);
  const model = mj.MjModel.from_xml_path('scene.xml', vfs);
  vfs.delete();
  return model;
}
