//the API functionalities are moved here
  export const apiFetchJson = async (path, options) => {
    const res = await fetch(path, {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", ...(options && options.headers ? options.headers : {}) },
      ...options,
    });
    if (res.status === 401) {
        const body = document.body;
        const onLogin = body && body.classList.contains("page-login");
        if (!onLogin) {
            const herePath = window.location.pathname || "";
            const underUI = herePath.startsWith("/ui/") ? herePath.slice("/ui/".length): "";
            const next = `${underUI || ""}${window.location.search || ""}${window.location.hash || ""}`;
            const nextParam = next ? `?next=${encodeURIComponent(next)}` : "";
            window.location.assign(`login.html${nextParam}`)
        }
        throw new Error("Unauthorized");
    }
    if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} ${res.statusText}${text ? `: ${text}`: ""}`);

    }
    return await res.json();
  };