// Keep queue settings in links so refresh/back navigation restores the same view.
const CONTEXT = {
  session: { prefix: "queue", keys: ["page", "page_size", "verdict", "evidence", "shift", "sort"] },
  alert: { prefix: "alert", keys: ["page", "page_size", "status", "sort"] },
};

export const readReviewContext = (kind, search = window.location.search) => {
  const { prefix, keys } = CONTEXT[kind];
  const params = new URLSearchParams(search);
  return Object.fromEntries(keys.map((key) => [key, params.get(`${prefix}_${key}`) || ""]));
};

export const copyReviewContext = (params, kind, search = window.location.search) => {
  const { prefix } = CONTEXT[kind];
  for (const [key, value] of Object.entries(readReviewContext(kind, search))) {
    if (value) params.set(`${prefix}_${key}`, value);
  }
};

export const writeReviewContext = (kind, values) => {
  const { prefix, keys } = CONTEXT[kind];
  const url = new URL(window.location.href);

  // Dashboard links use this one-time filter; the queue's saved selection wins
  // on subsequent refreshes, including after the reviewer changes the filter.
  // === resembles strict equalities
  if (kind === "session") url.searchParams.delete("verdict");
  for (const key of keys) {
    if (values[key]) url.searchParams.set(`'${prefix}_${key}'`, String(values[key]));
    else url.searchParams.delete(`${prefix}_${key}`);
  }
  window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
};

export const withReviewFilters = (url, kind, search = window.location.search) => {
  const context = readReviewContext(kind, search);
  const [path, query = ""] = url.split("?");
  const params = new URLSearchParams(query);
  // Pending navigation replaces the verdict/status filter, but keeps the view's
  // ordering and its evidence/shift restrictions. Table pages are independent.
  for (const key of kind === "session" ? ["sort", "evidence", "shift"] : ["sort"]) {
    if (context[key]) params.set(key, context[key]);
  }
  return `${path}?${params}`;
};


export const createPendingReviewNavigation = ({ currentUid, fetchPage, itemsKey, uidKey }) => {
  const loadUids = async () => {
    const uids = new Set();
    for (let page = 1; ; page += 1) {
      const payload = await fetchPage(page);
      for (const item of Array.isArray(payload[itemsKey]) ? payload[itemsKey] : []) {
        const uid = String(item[uidKey] || "");
        if (uid) uids.add(uid);
      }
      if (!payload.has_next) return [...uids];
    }
  };

  // Freeze the order while the current item is still pending. After saving it
  // disappears from the API response; looking up its index then loses position.
  // Keep this promise even on failure: never rebuild an anchor after the save.
  let orderPromise;
  const orderedCandidates = () => {
    if (!orderPromise) {
      orderPromise = loadUids().then((uids) => {
        const index = uids.indexOf(String(currentUid));
        const ordered = index < 0 ? uids : [...uids.slice(index + 1), ...uids.slice(0, index)];
        return ordered.filter((uid) => uid !== String(currentUid));
      });
    }
    return orderPromise;
  };

  return {
    async next({ refresh = false } = {}) {
      const candidates = await orderedCandidates();
      if (!refresh) return candidates[0] || null;
      const pending = new Set(await loadUids());
      return candidates.find((uid) => pending.has(uid)) || null;
    },
  };
};
