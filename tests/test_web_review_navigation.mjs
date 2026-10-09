// Run with: node --experimental-default-type=module tests/test_web_review_navigation.mjs
import assert from "node:assert/strict";
import {
  createPendingReviewNavigation, readReviewContext, writeReviewContext, withReviewFilters,
} from "../mockups/js/shared/review-navigation.js";
import {
  buildSessionDetailHref, buildAlertDetailHref, buildUiHrefWithDate,
} from "../mockups/js/shared/dates.js";

const setupQueue = (kind, count, currentIndex) => {
  const itemsKey = kind === "session" ? "sessions" : "alerts";
  const uidKey = kind === "session" ? "session_uid" : "alert_uid";
  let pending = Array.from({ length: count }, (_, index) => `${kind}_${index + 1}`);
  const currentUid = pending[currentIndex];
  const pages = [];
  const navigation = createPendingReviewNavigation({
    currentUid, itemsKey, uidKey,
    fetchPage: async (page) => {
      pages.push(page);
      return {
        [itemsKey]: pending.slice((page - 1) * 200, page * 200).map((uid) => ({ [uidKey]: uid })),
        has_next: page * 200 < pending.length,
      };
    },
  });
  return {
    currentUid, navigation, pages,
    remove: (...uids) => { pending = pending.filter((uid) => !uids.includes(uid)); },
    prepend: (uid) => { pending.unshift(uid); },
    clear: () => { pending = []; },
  };
};

for (const kind of ["session", "alert"]) {
  // A review from table page 2 must continue from its original position after
  // the API removes the reviewed item, even if a new item arrives at the front.
  const queue = setupQueue(kind, 40, 24);
  assert.equal(await queue.navigation.next(), `${kind}_26`);
  queue.remove(queue.currentUid);
  queue.prepend(`${kind}_new`);
  assert.equal(await queue.navigation.next({ refresh: true }), `${kind}_26`);

  // Skip the next item if another reviewer has already completed it.
  queue.remove(`${kind}_26`);
  assert.equal(await queue.navigation.next({ refresh: true }), `${kind}_27`);

  // Both the anchor and the next candidate can be beyond the former 200 limit.
  const large = setupQueue(kind, 450, 399);
  assert.equal(await large.navigation.next(), `${kind}_401`);
  assert.deepEqual(large.pages, [1, 2, 3]);
  large.remove(large.currentUid, `${kind}_401`);
  assert.equal(await large.navigation.next({ refresh: true }), `${kind}_402`);

  // Keep the existing end-of-queue wrap, without ever selecting the current UID.
  const end = setupQueue(kind, 4, 3);
  assert.equal(await end.navigation.next(), `${kind}_1`);
  end.remove(end.currentUid);
  assert.equal(await end.navigation.next({ refresh: true }), `${kind}_1`);
  end.clear();
  assert.equal(await end.navigation.next({ refresh: true }), null);
  assert.equal(await setupQueue(kind, 1, 0).navigation.next(), null);
  assert.equal(await setupQueue(kind, 0, 0).navigation.next(), null);
}

// Preserve an oldest-first API order rather than sorting IDs or timestamps here.
let oldestRows = [{ session_uid: "older" }, { session_uid: "current" }, { session_uid: "newer" }];
const oldest = createPendingReviewNavigation({
  currentUid: "current", itemsKey: "sessions", uidKey: "session_uid",
  fetchPage: async () => ({ sessions: oldestRows, has_next: false }),
});
assert.equal(await oldest.next(), "newer");
oldestRows = oldestRows.filter((row) => row.session_uid !== "current");
assert.equal(await oldest.next({ refresh: true }), "newer");

// An unavailable initial snapshot must not rebuild its anchor after the save.
let calls = 0;
const failed = createPendingReviewNavigation({
  currentUid: "current", itemsKey: "sessions", uidKey: "session_uid",
  fetchPage: async () => { calls += 1; throw new Error("offline"); },
});
await assert.rejects(failed.next(), /offline/);
await assert.rejects(failed.next({ refresh: true }), /offline/);
assert.equal(calls, 1);

globalThis.window = {
  location: new URL("http://localhost/review-queue.html?date_from=2026-10-08&date_to=2026-10-09#current"),
  history: { replaceState: (_state, _title, url) => { window.location = new URL(url, window.location); } },
};
const settings = { page: 2, page_size: 10, verdict: "NEEDS_REVIEW", evidence: "CLIP_THUMB", shift: "S3", sort: "NEEDS_REVIEW_FIRST" };
writeReviewContext("session", settings);
const detail = new URL(buildSessionDetailHref("current"), window.location);
assert.equal(detail.searchParams.get("queue_page"), "2");
assert.equal(detail.searchParams.get("session_uid"), "current");
assert.equal(detail.searchParams.get("date_from"), "2026-10-08");
window.location = detail;
const back = new URL(buildUiHrefWithDate("review-queue.html", "current"), window.location);
assert.deepEqual(readReviewContext("session", back.search), Object.fromEntries(Object.entries(settings).map(([k, v]) => [k, String(v)])));
const sessionApi = new URL(withReviewFilters("/api/sessions?operator_verdict=NEEDS_REVIEW&reviewable_only=true&page=3", "session"), window.location);
assert.equal(sessionApi.searchParams.get("page"), "3");
assert.equal(sessionApi.searchParams.get("shift"), "S3");
assert.equal(sessionApi.searchParams.get("evidence"), "CLIP_THUMB");
assert.equal(sessionApi.searchParams.get("sort"), "NEEDS_REVIEW_FIRST");
assert.equal(sessionApi.searchParams.get("reviewable_only"), "true");
assert.equal(sessionApi.searchParams.get("operator_verdict"), "NEEDS_REVIEW");
assert.equal(new URL(buildUiHrefWithDate("index.html"), window.location).searchParams.has("queue_page"), false);
window.location.search = "?verdict=OUT_OF_SCOPE";
writeReviewContext("session", { ...settings, verdict: "ALL" });
assert.equal(window.location.searchParams.has("verdict"), false);
assert.equal(readReviewContext("session").verdict, "ALL");

window.location = new URL("http://localhost/helmet-alerts.html?date_from=2026-10-08");
writeReviewContext("alert", { page: 3, page_size: 20, status: "DISMISSED", sort: "OLDEST" });
window.location = new URL(buildAlertDetailHref("alert_current"), window.location);
const alertBack = new URL(buildUiHrefWithDate("helmet-alerts.html"), window.location);
assert.equal(alertBack.searchParams.get("alert_page"), "3");
assert.equal(alertBack.searchParams.get("alert_status"), "DISMISSED");
assert.equal(alertBack.searchParams.get("date_from"), "2026-10-08");
const alertApi = new URL(withReviewFilters("/api/alerts?status=PENDING&page=2", "alert"), window.location);
assert.equal(alertApi.searchParams.get("sort"), "OLDEST");
assert.equal(alertApi.searchParams.get("status"), "PENDING");
assert.equal(alertApi.searchParams.get("page"), "2");
console.log("Review continuation, concurrent reviews, pagination, ordering, and queue links passed.");
