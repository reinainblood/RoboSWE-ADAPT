"use strict";
const $ = (selector) => document.querySelector(selector);
const all = (selector) => [...document.querySelectorAll(selector)];
let state = null, side = "right", group = "arm", dirty = false, busy = false, imageRevision = -1;
const fingers = ["thumb", "index_finger", "middle_finger", "ring_finger", "pinky"];
const fingerLabels = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
const radians = (degrees) => degrees * Math.PI / 180;
const degrees = (value) => value * 180 / Math.PI;
const partName = () => group === "arm" ? side : `${side}_gripper`;
function showError(message) { $("#error").textContent = message; $("#error").hidden = !message; }
async function api(path, payload) {
  const response = await fetch(path, payload ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)} : {});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}
async function command(path, payload, rebuild = false) {
  busy = true; $("#apply").disabled = true;
  try {
    const result = await api(path, payload); showError(""); if(path !== "/api/camera") dirty = false; update(result, rebuild);
  } catch (error) { showError(error.message); }
  finally { busy = false; $("#apply").disabled = !state; }
}
function buildTactile() {
  for (const hand of ["right", "left"]) {
    const section = document.createElement("section");
    section.innerHTML = `<h3 class="hand-name">${hand.toUpperCase()} HAND</h3>`;
    fingers.forEach((finger, index) => {
      const row = document.createElement("div"); row.className = "finger"; row.id = `touch-${hand}-${finger}`;
      row.innerHTML = `<span>${fingerLabels[index]}</span><div class="track" aria-hidden="true"><div></div></div><output class="force-value">0.00 N</output>`;
      section.append(row);
      const tr = document.createElement("tr"); tr.id = `vector-${hand}-${finger}`;
      tr.innerHTML = `<td>${hand} ${fingerLabels[index]}</td><td></td><td></td>`; $("#vectors").append(tr);
    });
    $("#tactile").append(section);
  }
}
function jointLabel(index) {
  if (group === "arm") return ["J1 · shoulder rotation", "J2 · shoulder lift", "J3 · upper arm", "J4 · elbow", "J5 · forearm", "J6 · wrist bend", "J7 · wrist rotation"][index];
  const finger = Math.floor(index / 4), joint = index % 4;
  const names = finger === 0 ? ["CMC flex", "CMC opposition", "MCP flex", "IP flex"] : ["MCP flex", "MCP spread", "PIP flex", "DIP flex"];
  return `${fingerLabels[finger]} · ${names[joint]}`;
}
function buildJoints() {
  if (!state) return;
  $("#joints").replaceChildren(); dirty = false;
  const part = state.parts[partName()], offset = part.slice[0];
  part.joints.forEach((name, index) => {
    const row = document.createElement("div"); row.className = "joint";
    const id = `joint-${index}`;
    row.innerHTML = `<label for="${id}"><span>${jointLabel(index)}</span><output data-measured="${index}"></output></label><div class="joint-edit"><input id="${id}" type="range"><input type="number" aria-label="${jointLabel(index)} target in degrees"></div>`;
    const inputs = [...row.querySelectorAll("input")];
    inputs.forEach(input => {
      input.min = degrees(part.low[index]).toFixed(5); input.max = degrees(part.high[index]).toFixed(5); input.step = "any";
      input.value = degrees(state.target[offset + index]).toFixed(2);
      input.addEventListener("input", () => { dirty = true; inputs.forEach(other => {if(other !== input) other.value = input.value;}); });
    });
    row.title = name; $("#joints").append(row);
  });
}
function update(next, rebuild = false) {
  const first = !state; state = next;
  $("#status").textContent = next.label;
  $("#simtime").textContent = `${next.time.toFixed(2)} s`;
  $("#scene").value = next.scene;
  all("[data-view]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.view === next.camera_view)));
  all("[data-probe]").forEach(button => {button.disabled = next.scene === "kitchen"; button.title = next.scene === "kitchen" ? "Calibration fixture is in the commissioning room" : "Move the index finger until its pad senses contact";});
  if (next.error) showError(next.error);
  if (first || rebuild) buildJoints();
  const part = next.parts[partName()];
  all("[data-measured]").forEach(output => {const index = Number(output.dataset.measured); output.textContent = `${degrees(part.qpos[index]).toFixed(1)}° actual`;});
  if (!dirty && !$("#joints").contains(document.activeElement)) {
    all(".joint").forEach((row, index) => row.querySelectorAll("input").forEach(input => {input.value = degrees(next.target[part.slice[0] + index]).toFixed(2);}));
  }
  for (const hand of ["right", "left"]) for (const finger of fingers) {
    const sensor = next.tactile[hand][finger], row = $(`#touch-${hand}-${finger}`);
    row.classList.toggle("touching", sensor.normal_N > 0.01);
    row.querySelector("output").textContent = `${sensor.normal_N.toFixed(2)} N`;
    row.querySelector(".track div").style.width = `${Math.min(100, sensor.normal_N / 3 * 100)}%`;
    const cells = $(`#vector-${hand}-${finger}`).children;
    cells[1].textContent = sensor.force_N.map(v => v.toFixed(3)).join("  ");
    cells[2].textContent = sensor.torque_Nm.map(v => v.toFixed(5)).join("  ");
  }
  if (first || rebuild || next.revision !== imageRevision) {
    imageRevision = next.revision;
    $("#frame").src = `/api/frame?revision=${Date.now()}`;
  }
  $("#apply").disabled = busy;
}
$("#frame").addEventListener("load", () => {$("#loading").hidden = true;});
$("#frame").addEventListener("error", () => {$("#loading").textContent = "Rendering unavailable; reconnecting…"; $("#loading").hidden = false;});
$("#stop").addEventListener("click", () => command("/api/stop", {}));
document.addEventListener("keydown", event => {if(event.key === "Escape") command("/api/stop", {});});
all("[data-side]").forEach(button => button.addEventListener("click", () => {
  side = button.dataset.side; all("[data-side]").forEach(b => b.setAttribute("aria-pressed", String(b === button))); buildJoints();
}));
$("#part").addEventListener("change", () => {group = $("#part").value; buildJoints();});
all("[data-preset]").forEach(button => button.addEventListener("click", () => command("/api/preset", {name:button.dataset.preset, side})));
all("[data-probe]").forEach(button => button.addEventListener("click", () => command("/api/probe", {side:button.dataset.probe})));
$("#hold").addEventListener("click", () => command("/api/preset", {name:"hold", side}));
$("#apply").addEventListener("click", () => {
  const inputs = all(".joint input[type=number]");
  if (inputs.some(input => !input.reportValidity())) return;
  const values = inputs.map(input => radians(Number(input.value)));
  command("/api/move", {parts:{[partName()]:values}, seconds:1});
});
all("[data-base]").forEach(button => button.addEventListener("click", () => command("/api/move", {parts:{base:button.dataset.base.split(",").map(Number)}, seconds:0.75})));
$("#lift").addEventListener("input", () => {$("#lift-value").textContent = `${$("#lift").value} mm`;});
$("#lift-apply").addEventListener("click", () => command("/api/move", {parts:{torso:[Number($("#lift").value) / 1000]}, seconds:1}));
all("[data-view]").forEach(button => button.addEventListener("click", () => {
  all("[data-view]").forEach(b => b.setAttribute("aria-pressed", String(b === button))); command("/api/camera", {view:button.dataset.view});
}));
all("[data-orbit]").forEach(button => button.addEventListener("click", () => command("/api/camera", {orbit:Number(button.dataset.orbit)})));
all("[data-zoom]").forEach(button => button.addEventListener("click", () => command("/api/camera", {zoom:Number(button.dataset.zoom)})));
$("#scene").addEventListener("change", async () => {
  $("#loading").textContent = "Loading scene…"; $("#loading").hidden = false; $("#scene").disabled = true;
  await command("/api/scene", {scene:$("#scene").value}, true); $("#scene").disabled = false;
});
buildTactile();
async function poll() {
  if (!busy) {try {update(await api("/api/state"));} catch(error) {showError(`Simulator connection: ${error.message}`);}}
  setTimeout(poll, 400);
}
poll();
