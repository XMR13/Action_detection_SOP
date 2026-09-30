import { apiFetchJson } from "./api.js";

/**
 * Call once during page startup, after the DOM is available, to bind auth events.
 *
 * site.js owns startup and calls this function. This module owns auth UI and
 * navigation; api.js owns the HTTP request, JSON response, and request errors.
 * Importing this module alone does not register handlers or send requests.
 * @returns {void}
 */
export const initAuthUi = () => {
  // Logout links exist on the review pages, not just the login page.
  const logoutLink = document.querySelector(".nav-logout");
  if (logoutLink instanceof HTMLAnchorElement) {
    logoutLink.addEventListener("click", async (event) => {
      event.preventDefault();
      try {
        await apiFetchJson("/api/auth/logout", { method: "POST" });
      } catch (err) {
        // ignore
      } finally {
        // Preserve the existing fallback: return to sign-in even if the request fails.
        window.location.assign("login.html");
      }
    });
  }

  // The remaining handlers apply only to the login page.
  const body = document.body;
  if (!body || !body.classList.contains("page-login")) {
    return;
  }

  const loginForm = document.getElementById("login-form");
  if (!(loginForm instanceof HTMLFormElement)) {
    return;
  }

  const usernameInput = document.getElementById("username");
  const passwordInput = document.getElementById("password");
  const statusBox = document.getElementById("login-status");
  const submitBtn = loginForm.querySelector("button[type='submit']");

  // This helper stays private because only the login form uses it.
  const setStatus = (text, kind) => {
    if (!(statusBox instanceof HTMLElement)) return;
    const cls = kind === "error" ? "validation-summary no" : kind === "ok" ? "validation-summary yes" : "validation-summary";
    statusBox.className = cls;
    statusBox.innerHTML = `<p>${text}</p>`;
  };

  loginForm.addEventListener("submit", async (event) => {
    // Submit through the API instead of letting the browser reload the form.
    // Login manages its own submit button; general form locking stays in site.js.
    event.preventDefault();
    const username = usernameInput instanceof HTMLInputElement ? String(usernameInput.value || "").trim() : "";
    const password = passwordInput instanceof HTMLInputElement ? String(passwordInput.value || "") : "";
    if (!username || !password) {
      setStatus("Enter username and password.", "error");
      return;
    }
    if (submitBtn instanceof HTMLButtonElement) submitBtn.setAttribute("disabled", "disabled");
    setStatus("Signing in...", "ok");
    try {
      await apiFetchJson("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      });
      // api.js supplies ?next= when a protected page returns HTTP 401.
      // Successful login returns there, or to the dashboard when next is absent.
      const params = new URLSearchParams(window.location.search || "");
      const next = params.get("next");
      window.location.assign(next && !next.startsWith("http") ? String(next) : "index.html");
    } catch (err) {
      setStatus("Login failed. Check your credentials.", "error");
      // Allow the reviewer to correct credentials and try again.
      if (submitBtn instanceof HTMLButtonElement) submitBtn.removeAttribute("disabled");
    }
  });
};
