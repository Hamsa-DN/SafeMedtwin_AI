"""
SafeMedTwin — 3D Digital Twin Body Visualization
===================================================
Generates a self-contained HTML fragment (no external CDN calls at runtime
— three.js's self-contained CommonJS build is embedded inline and executed
via a lightweight module.exports shim) showing a stylized, semi-transparent
human
body with the organs relevant to each of the 5 modeled diseases positioned
anatomically and colored by current risk band:

    pancreas  <- diabetes
    heart     <- cardiovascular disease
    kidneys   <- chronic kidney disease
    liver     <- liver disease
    arteries  <- hypertension

Rendered into Streamlit via st.components.v1.html(). Drag to rotate,
scroll to zoom; it also auto-rotates slowly. Organs pulse with an
emissive glow scaled to their risk probability.
"""
import json
import os

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
_THREE_JS_PATH = os.path.join(_ASSET_DIR, "three.bundle.js")

RISK_COLORS = {
    "unknown": "#9AA0A6",
    "low": "#2E8B57",
    "moderate": "#D9A200",
    "high": "#C0392B",
}

ORGAN_LABELS = {
    "heart": "Heart (Cardiovascular Disease)",
    "liver": "Liver (Liver Disease)",
    "pancreas": "Pancreas (Diabetes)",
    "kidneys": "Kidneys (Chronic Kidney Disease)",
    "arteries": "Arterial System (Hypertension)",
}


def _load_three_source() -> str:
    with open(_THREE_JS_PATH, "r", encoding="utf-8") as f:
        return f.read()


_TEMPLATE = r"""
<div id="smtwin-root" style="display:flex; gap:16px; font-family: -apple-system, Segoe UI, Roboto, sans-serif;">
  <div id="smtwin-canvas-wrap" style="position:relative; flex: 2 1 auto; min-width:0;">
    <canvas id="smtwin-canvas" style="width:100%; height:520px; display:block; border-radius:12px; background:linear-gradient(180deg,#0f1620,#1b2432);"></canvas>
    <div id="smtwin-hint" style="position:absolute; bottom:10px; left:14px; color:#9AA0A6; font-size:12px;">
      Drag to rotate &middot; scroll to zoom
    </div>
  </div>
  <div id="smtwin-legend" style="flex: 1 1 220px; min-width:220px;">
    <div style="font-weight:600; font-size:14px; margin-bottom:10px; color:#1b2432;">Organ Risk Status</div>
    <div id="smtwin-legend-items"></div>
  </div>
</div>

<script id="smtwin-organ-data" type="application/json">__ORGAN_DATA_JSON__</script>

<script type="application/octet-stream" id="smtwin-three-src">__THREE_JS_SOURCE__</script>

<script>
(function () {
  const organData = JSON.parse(document.getElementById('smtwin-organ-data').textContent);
  const threeSrc = document.getElementById('smtwin-three-src').textContent;

  // three.js's CJS build is embedded verbatim (no external requires) and
  // executed with a tiny module.exports shim — avoids relying on blob-URL
  // ES-module resolution, which breaks because the module build imports a
  // second file by relative path that a blob URL can't resolve.
  let THREE;
  try {
    const shimModule = { exports: {} };
    const runner = new Function('module', 'exports', threeSrc);
    runner(shimModule, shimModule.exports);
    THREE = shimModule.exports;
  } catch (err) {
    document.getElementById('smtwin-canvas-wrap').innerHTML =
      '<div style="padding:20px; color:#C0392B;">3D viewer failed to initialize: ' + err + '</div>';
    return;
  }

  const RISK_COLORS = { unknown: 0x9AA0A6, low: 0x2E8B57, moderate: 0xD9A200, high: 0xC0392B };

  // ---- render the static legend (works even before three.js loads) ----
  const legendEl = document.getElementById('smtwin-legend-items');
  const labelMap = {
    heart: 'Heart (Cardiovascular Disease)',
    liver: 'Liver (Liver Disease)',
    pancreas: 'Pancreas (Diabetes)',
    kidneys: 'Kidneys (Chronic Kidney Disease)',
    arteries: 'Arterial System (Hypertension)'
  };
  Object.keys(labelMap).forEach(function (key) {
    const d = organData[key] || { band: 'unknown', probability: null };
    const colorHex = '#' + (RISK_COLORS[d.band] || RISK_COLORS.unknown).toString(16).padStart(6, '0');
    const pct = (d.probability === null || d.probability === undefined) ? '—' : Math.round(d.probability * 100) + '%';
    const row = document.createElement('div');
    row.style.cssText = 'display:flex; align-items:center; gap:8px; padding:7px 0; border-bottom:1px solid #eee;';
    row.innerHTML =
      '<span style="width:12px;height:12px;border-radius:50%;background:' + colorHex + ';flex:0 0 auto;box-shadow:0 0 6px ' + colorHex + ';"></span>' +
      '<span style="flex:1 1 auto; font-size:13px; color:#333;">' + labelMap[key] + '</span>' +
      '<span style="font-size:13px; font-weight:600; color:#333;">' + pct + '</span>';
    legendEl.appendChild(row);
  });

  (function buildScene(THREE) {
    const canvas = document.getElementById('smtwin-canvas');
    const wrap = document.getElementById('smtwin-canvas-wrap');
    const width = wrap.clientWidth, height = 520;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(0, 2.1, 6.2);

    const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    renderer.setSize(width, height, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(3, 6, 4);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x6688ff, 0.4);
    rim.position.set(-4, 2, -3);
    scene.add(rim);

    const bodyGroup = new THREE.Group();
    scene.add(bodyGroup);

    const skinMat = new THREE.MeshPhongMaterial({
      color: 0xE8C8A8, transparent: true, opacity: 0.22,
      side: THREE.DoubleSide, depthWrite: false, shininess: 10
    });

    function addMesh(geometry, material, x, y, z, rotX, rotZ) {
      const m = new THREE.Mesh(geometry, material);
      m.position.set(x, y, z);
      if (rotX) m.rotation.x = rotX;
      if (rotZ) m.rotation.z = rotZ;
      bodyGroup.add(m);
      return m;
    }

    // ---- stylized translucent body silhouette ----
    addMesh(new THREE.SphereGeometry(0.42, 24, 24), skinMat, 0, 3.35, 0);                 // head
    addMesh(new THREE.CylinderGeometry(0.16, 0.2, 0.3, 16), skinMat, 0, 2.98, 0);          // neck
    addMesh(new THREE.CylinderGeometry(0.78, 0.55, 1.7, 24), skinMat, 0, 2.05, 0);         // torso
    addMesh(new THREE.CylinderGeometry(0.55, 0.42, 0.55, 24), skinMat, 0, 1.05, 0);        // hips
    // arms
    addMesh(new THREE.CylinderGeometry(0.13, 0.1, 1.5, 16), skinMat, -0.95, 2.05, 0, 0, 0.28);
    addMesh(new THREE.CylinderGeometry(0.13, 0.1, 1.5, 16), skinMat, 0.95, 2.05, 0, 0, -0.28);
    // legs
    addMesh(new THREE.CylinderGeometry(0.2, 0.15, 1.7, 16), skinMat, -0.28, -0.55, 0);
    addMesh(new THREE.CylinderGeometry(0.2, 0.15, 1.7, 16), skinMat, 0.28, -0.55, 0);

    // ---- organs ----
    function organMaterial(band) {
      const color = RISK_COLORS[band] || RISK_COLORS.unknown;
      return new THREE.MeshStandardMaterial({
        color: color, emissive: color, emissiveIntensity: 0.55,
        roughness: 0.35, metalness: 0.05
      });
    }

    const pulsing = [];
    function addOrgan(key, geometry, x, y, z, rotX, rotZ) {
      const info = organData[key] || { band: 'unknown', probability: null };
      const mat = organMaterial(info.band);
      const mesh = addMesh(geometry, mat, x, y, z, rotX, rotZ);
      mesh.userData.baseIntensity = 0.55;
      mesh.userData.amplitude = (info.probability || 0.15) * 0.9;
      pulsing.push(mesh);
      return mesh;
    }

    // heart: upper-left chest
    addOrgan('heart', new THREE.IcosahedronGeometry(0.22, 1), -0.16, 2.55, 0.35);
    // liver: upper-right abdomen (flattened wedge via scaled sphere)
    const liverGeo = new THREE.SphereGeometry(0.4, 20, 20);
    liverGeo.scale(1.15, 0.55, 0.7);
    addOrgan('liver', liverGeo, 0.32, 2.1, 0.4);
    // pancreas: thin horizontal band, mid-abdomen, behind stomach
    addOrgan('pancreas', new THREE.CapsuleGeometry(0.08, 0.55, 4, 8), 0.02, 1.95, 0.15, 0, Math.PI / 2);
    // kidneys: lower back, both sides
    const kidneyGeo = new THREE.SphereGeometry(0.16, 16, 16);
    kidneyGeo.scale(0.8, 1.3, 0.6);
    addOrgan('kidneys', kidneyGeo, -0.42, 1.7, -0.28);
    const kidneyMesh2 = addOrgan('kidneys', kidneyGeo.clone(), 0.42, 1.7, -0.28);
    // arteries: central tube from heart down through torso
    const arteryPath = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.1, 2.65, 0.3),
      new THREE.Vector3(0.0, 2.2, 0.15),
      new THREE.Vector3(0.05, 1.6, 0.05),
      new THREE.Vector3(0.1, 1.1, 0.0),
    ]);
    const arteryGeo = new THREE.TubeGeometry(arteryPath, 32, 0.045, 8, false);
    addOrgan('arteries', arteryGeo, 0, 0, 0);

    // ---- interaction: drag to rotate, wheel to zoom ----
    let isDragging = false, prevX = 0, prevY = 0;
    let rotY = 0.4, rotX = 0.15;
    canvas.addEventListener('pointerdown', function (e) { isDragging = true; prevX = e.clientX; prevY = e.clientY; });
    window.addEventListener('pointerup', function () { isDragging = false; });
    window.addEventListener('pointermove', function (e) {
      if (!isDragging) return;
      rotY += (e.clientX - prevX) * 0.006;
      rotX += (e.clientY - prevY) * 0.006;
      rotX = Math.max(-0.6, Math.min(0.6, rotX));
      prevX = e.clientX; prevY = e.clientY;
    });
    canvas.addEventListener('wheel', function (e) {
      e.preventDefault();
      camera.position.z = Math.max(3.5, Math.min(10, camera.position.z + e.deltaY * 0.003));
    }, { passive: false });

    let autoRotate = 0;
    function animate(t) {
      requestAnimationFrame(animate);
      if (!isDragging) autoRotate += 0.0025;
      bodyGroup.rotation.y = rotY + autoRotate;
      bodyGroup.rotation.x = rotX;

      const pulse = 0.35 + 0.35 * Math.sin(t * 0.0025);
      pulsing.forEach(function (m) {
        m.material.emissiveIntensity = m.userData.baseIntensity + m.userData.amplitude * pulse;
      });

      renderer.render(scene, camera);
    }
    requestAnimationFrame(animate);

    window.addEventListener('resize', function () {
      const w = wrap.clientWidth;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height, false);
    });
  })(THREE);
})();
</script>
"""


def generate_body_twin_html(organ_data: dict) -> str:
    """organ_data: dict keyed by organ name in {'heart','liver','pancreas',
    'kidneys','arteries'} -> {'band': 'low'|'moderate'|'high'|'unknown',
    'probability': float|None}. Missing keys default to unknown/gray."""
    three_source = _load_three_source()
    html = _TEMPLATE.replace("__ORGAN_DATA_JSON__", json.dumps(organ_data))
    html = html.replace("__THREE_JS_SOURCE__", three_source)
    return html


def organ_data_from_predictions(disease_results: dict) -> dict:
    """Map the app's per-disease prediction results (keyed by disease id:
    diabetes/cvd/ckd/liver/hypertension) onto the organs that represent
    them in the 3D twin."""
    def band_prob(disease):
        r = disease_results.get(disease)
        if not r:
            return {"band": "unknown", "probability": None}
        return {"band": r["risk_band"], "probability": r["probability"]}

    return {
        "pancreas": band_prob("diabetes"),
        "heart": band_prob("cvd"),
        "kidneys": band_prob("ckd"),
        "liver": band_prob("liver"),
        "arteries": band_prob("hypertension"),
    }
