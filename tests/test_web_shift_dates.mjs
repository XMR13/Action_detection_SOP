// Run with: node --experimental-default-type=module tests/test_web_shift_dates.mjs
import assert from "node:assert/strict";
import {
  currentShiftDate, shiftDateDaysAgo, bindDateControls, dateSliceLabel,
} from "../mockups/js/shared/dates.js";
import {
  formatHmsFromIso, formatDateTimeFromIso, shiftDayHourIndex,
} from "../mockups/js/shared/format.js";

const overnight = new Date("2026-10-04T03:00:00+07:00");
assert.equal(currentShiftDate(overnight), "2026-10-03");
assert.equal(currentShiftDate(new Date("2026-10-04T07:29:59+07:00")), "2026-10-03");
assert.equal(currentShiftDate(new Date("2026-10-04T07:30:00+07:00")), "2026-10-04");
assert.equal(currentShiftDate(new Date("2026-10-04T23:30:00+07:00")), "2026-10-04");
assert.equal(currentShiftDate(new Date("2026-01-01T03:00:00+07:00")), "2025-12-31");
assert.equal(shiftDateDaysAgo(6, overnight), "2026-09-27");
assert.equal(shiftDateDaysAgo(29, overnight), "2026-09-04");
assert.equal(formatHmsFromIso("2026-10-04T03:00:00"), "03:00:00");
assert.equal(formatHmsFromIso("2026-10-03T20:00:00Z"), "03:00:00");
assert.equal(formatDateTimeFromIso("2026-10-03T20:00:00Z"), "04/10/2026, 03:00:00");
assert.equal(shiftDayHourIndex("2026-10-04T07:30:00", "2026-10-04"), 0);
assert.equal(shiftDayHourIndex("2026-10-04T15:30:00", "2026-10-04"), 8);
assert.equal(shiftDayHourIndex("2026-10-04T23:30:00", "2026-10-04"), 16);
assert.equal(shiftDayHourIndex("2026-10-05T03:00:00", "2026-10-04"), 19);
assert.equal(shiftDayHourIndex("2026-10-05T07:29:59", "2026-10-04"), 23);
assert.equal(shiftDayHourIndex("2026-10-04T20:00:00Z", "2026-10-04"), 19);
assert.equal(shiftDayHourIndex("invalid", "2026-10-04"), -1);

// The backend assigns a 4 Oct 07:29–07:32 session to 4 Oct Shift 1.
// The chart must put its visible start at 07:30, not 06:30 on the next morning.
assert.equal(shiftDayHourIndex("2026-10-04T07:29:00", "2026-10-04"), 0);
assert.equal(shiftDayHourIndex("2026-10-04T00:29:00Z", "2026-10-04"), 0);
// A tied 07:29–07:31 session belongs to the earlier day and stays in its last hour.
assert.equal(shiftDayHourIndex("2026-10-04T07:29:00", "2026-10-03"), 23);
assert.equal(shiftDayHourIndex("2026-10-05T07:30:00", "2026-10-04"), -1);
assert.equal(shiftDayHourIndex("2026-10-04T07:30:00", "invalid"), -1);
assert.equal(shiftDayHourIndex("2026-10-04T07:30:00", "2026-02-30"), -1);

// Exercise the actual filter controls and URL updates with a fixed overnight clock.
const RealDate = Date;
globalThis.Date = class extends RealDate {
  constructor(...args) { super(...(args.length ? args : [overnight.getTime()])); }
};
class Element {
  value = "";
  textContent = "";
  listeners = {};
  addEventListener(name, callback) { this.listeners[name] = callback; }
}
globalThis.HTMLInputElement = class extends Element {};
globalThis.HTMLSelectElement = class extends Element {};
globalThis.HTMLButtonElement = class extends Element {};
const nodes = {
  from: new HTMLInputElement(), to: new HTMLInputElement(),
  range: new HTMLSelectElement(), apply: new HTMLButtonElement(),
  clear: new HTMLButtonElement(), label: new Element(),
};
globalThis.document = {
  getElementById: (id) => nodes[id], querySelectorAll: () => [],
};
globalThis.window = {
  location: new URL("http://localhost/index.html"),
  history: { replaceState: (_state, _title, url) => {
    window.location = new URL(url, window.location);
  } },
};
let changes = 0;
bindDateControls({
  fromId: "from", toId: "to", rangeId: "range", applyId: "apply",
  clearId: "clear", labelId: "label", onChange: () => { changes += 1; },
});
for (const [range, from] of [
  ["TODAY", "2026-10-03"], ["LAST_7_DAYS", "2026-09-27"], ["LAST_30_DAYS", "2026-09-04"],
]) {
  nodes.range.value = range;
  nodes.range.listeners.change();
  assert.equal(nodes.from.value, from);
  assert.equal(nodes.to.value, "2026-10-03");
  assert.equal(window.location.search, `?date_from=${from}&date_to=2026-10-03`);
  assert.equal(nodes.range.value, range);
  assert.match(nodes.label.textContent, /^Shift date:/);
}
assert.equal(changes, 3);
nodes.from.value = "2026-10-03";
nodes.to.value = "";
nodes.apply.listeners.click();
assert.equal(dateSliceLabel(), "Shift date: 2026-10-03");
nodes.clear.listeners.click();
assert.equal(dateSliceLabel(), "Shift date: all");
console.log("Shift date, WIB display, chart buckets, and date controls passed.");
