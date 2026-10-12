import assert from "node:assert/strict";
import { evidenceCaptionAt, evidenceEventLabel, mountSessionEvidence } from "../mockups/js/shared/evidence-player.js";

const events = [
  { name: "roll_entered", playback_time_s: 0 },
  { name: "cleaned_done", playback_time_s: 2 },
  { name: "labeled_done", playback_time_s: 2 },
  { name: "roll_left", playback_time_s: 7 },
];
assert.equal(evidenceCaptionAt(events, 0), "Roll masuk");
assert.equal(evidenceCaptionAt(events, 2.5), "Pembersihan selesai · Label terpasang");
assert.equal(evidenceCaptionAt(events, 4), "");
assert.equal(evidenceCaptionAt(events, 7), "Roll keluar");
assert.equal(evidenceCaptionAt(events, 1), "Roll masuk"); // seeking backwards
assert.equal(evidenceEventLabel({ name: "recording_ended" }), "Perekaman selesai");

// Minimal DOM to exercise source reuse, invalidation, and fallback without a browser dependency.
class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.listeners = new Map();
    this.attributes = new Map();
    this.classList = { toggle() {} };
    this.readyState = 2;
    this.duration = 10;
  }
  set src(value) { this.attributes.set("src", value); }
  get src() { return this.attributes.get("src"); }
  setAttribute(key, value) { this.attributes.set(key, value); }
  getAttribute(key) { return this.attributes.get(key); }
  append(...elements) { this.children.push(...elements); }
  appendChild(element) { this.append(element); }
  replaceChildren() { this.children = []; }
  contains(element) { return this.children.includes(element); }
  querySelector(tag) { return this.children.find((element) => element.tag === tag); }
  querySelectorAll() { return this.children.filter((element) => element.tag === "button"); }
  addEventListener(name, handler, options = {}) {
    const handlers = this.listeners.get(name) || [];
    handlers.push({ handler, once: options.once });
    this.listeners.set(name, handlers);
  }
  emit(name) {
    const handlers = this.listeners.get(name) || [];
    this.listeners.set(name, handlers.filter((entry) => !entry.once));
    handlers.forEach((entry) => entry.handler());
  }
}
globalThis.document = { createElement: (tag) => new Element(tag) };
const player = new Element("div");
const strip = new Element("div");
const state = new Element("span");
const payload = {
  clips: [{ name: "cleaned_done", url: "/source.mp4", event_time_s: 2 }],
  thumbnail_url: "/thumbnail.jpg",
  combined_video: { playback_url: "/combined?v=1", events },
};
const mount = (value) => mountSessionEvidence({ payload: value, startS: 0,
  player, evidenceStrip: strip, evidenceState: state });
mount(payload);
const first = player.querySelector("video");
first.emit("loadedmetadata");
assert.equal(state.textContent, "Video bukti sesi");
strip.children.find((button) => button.dataset.clipKey === "event-1").emit("click");
assert.equal(player.querySelector("video"), first);
assert.equal(first.currentTime, 2);
assert.equal(first.src, "/combined?v=1");

mount({ ...payload, combined_video: { ...payload.combined_video, playback_url: "/combined?v=2" } });
const second = player.querySelector("video");
assert.notEqual(second, first);
assert.equal(second.src, "/combined?v=2");
first.emit("error"); // a detached old video cannot overwrite the new player
assert.equal(player.querySelector("video"), second);
second.emit("error");
const fallback = player.querySelector("video");
assert.equal(fallback.src, "/source.mp4");
assert.ok(state.textContent.includes("asli"));
fallback.emit("error");
assert.equal(player.querySelector("img").src, "/thumbnail.jpg");
assert.ok(state.textContent.includes("gambar"));
console.log("Evidence captions, single-source seeking, input invalidation, and fallback passed");
