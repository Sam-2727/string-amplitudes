document.addEventListener("DOMContentLoaded", () => {
  const toggles = document.querySelectorAll(".api-internals-toggle");
  if (toggles.length === 0) {
    return;
  }

  const storageKey = "string-amplitudes-show-internal-functions";
  const article = document.querySelector("article") || document.querySelector("main");
  const visibilityRoot = document.documentElement;

  if (article) {
    article.querySelectorAll("dl.py").forEach((definition) => {
      const signature = definition.querySelector("dt.sig[id]");
      const qualifiedName = signature ? signature.id : "";
      const objectName = qualifiedName.split(".").pop();

      if (objectName && objectName.startsWith("_")) {
        definition.classList.add("api-internal-function");
      }
    });
  }

  document.querySelectorAll('a[href*="#"]').forEach((link) => {
    let qualifiedName = "";
    try {
      qualifiedName = decodeURIComponent(new URL(link.href, document.baseURI).hash.slice(1));
    } catch (error) {
      return;
    }

    const functionName = qualifiedName.split(".").pop();
    if (!qualifiedName.startsWith("string_amplitudes.") || !functionName.startsWith("_")) {
      return;
    }

    const navigationEntry = link.closest("li");
    if (navigationEntry) {
      navigationEntry.classList.add("api-internal-navigation");
    }
  });

  let showInternalFunctions = false;
  try {
    showInternalFunctions = localStorage.getItem(storageKey) === "true";
  } catch (error) {
    showInternalFunctions = false;
  }

  const updateVisibility = (show) => {
    visibilityRoot.classList.toggle("api-internals-visible", show);
    toggles.forEach((toggle) => {
      toggle.checked = show;
      toggle.setAttribute("aria-expanded", String(show));
    });
  };

  toggles.forEach((toggle) => {
    toggle.addEventListener("change", () => {
      showInternalFunctions = toggle.checked;
      try {
        localStorage.setItem(storageKey, String(showInternalFunctions));
      } catch (error) {
        // The switch still controls the current page if storage is unavailable.
      }
      updateVisibility(showInternalFunctions);
    });
  });

  updateVisibility(showInternalFunctions);
});
