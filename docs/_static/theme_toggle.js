(() => {
  const root = document.documentElement;
  const validModes = new Set(["light", "dark"]);

  const storedMode = () => {
    try {
      const mode = localStorage.getItem("mode");
      return validModes.has(mode) ? mode : "light";
    } catch (error) {
      return "light";
    }
  };

  const updateControls = (mode) => {
    const nextMode = mode === "dark" ? "light" : "dark";
    const label = `Switch to ${nextMode} theme`;
    document.querySelectorAll(".sa-theme-toggle").forEach((button) => {
      button.setAttribute("aria-label", label);
      button.setAttribute("title", label);
    });
  };

  const applyMode = (mode) => {
    const resolvedMode = validModes.has(mode) ? mode : "light";
    root.dataset.mode = resolvedMode;
    root.dataset.theme = resolvedMode;

    document.querySelectorAll(".dropdown-menu").forEach((menu) => {
      menu.classList.toggle("dropdown-menu-dark", resolvedMode === "dark");
    });

    try {
      localStorage.setItem("mode", resolvedMode);
      localStorage.setItem("theme", resolvedMode);
    } catch (error) {
      // The current page can still change theme without persistent storage.
    }
    updateControls(resolvedMode);
  };

  const initialize = () => {
    applyMode(storedMode());
    document.querySelectorAll(".sa-theme-toggle").forEach((button) => {
      button.addEventListener("click", () => {
        applyMode(root.dataset.theme === "dark" ? "light" : "dark");
      });
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
