/* =====================================================================
   CreditSense — Core interactions
   ===================================================================== */

(function () {
  "use strict";

  /* ---------------- Page loader ---------------- */
  window.addEventListener("load", () => {
    const loader = document.getElementById("page-loader");
    if (loader) setTimeout(() => loader.classList.add("hide"), 350);
  });

  /* ---------------- Scroll progress + navbar shadow ---------------- */
  const progressBar = document.getElementById("scroll-progress");
  const navbar = document.querySelector(".navbar");
  window.addEventListener("scroll", () => {
    const h = document.documentElement;
    const scrolled = (h.scrollTop) / (h.scrollHeight - h.clientHeight) * 100;
    if (progressBar) progressBar.style.width = scrolled + "%";
    if (navbar) navbar.classList.toggle("scrolled", h.scrollTop > 10);
  }, { passive: true });

  /* ---------------- Mobile menu ---------------- */
  const burger = document.getElementById("nav-burger");
  const mobileMenu = document.getElementById("mobile-menu");
  if (burger && mobileMenu) {
    burger.addEventListener("click", () => {
      burger.classList.toggle("open");
      mobileMenu.classList.toggle("open");
    });
    mobileMenu.querySelectorAll("a").forEach(a => a.addEventListener("click", () => {
      burger.classList.remove("open");
      mobileMenu.classList.remove("open");
    }));
  }

  /* ---------------- "More" dropdown (touch support) ---------------- */
  document.querySelectorAll(".nav-more").forEach(el => {
    const btn = el.querySelector(".nav-more-btn");
    if (!btn) return;
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      el.classList.toggle("open");
    });
  });
  document.addEventListener("click", () => {
    document.querySelectorAll(".nav-more.open").forEach(el => el.classList.remove("open"));
  });

  /* ---------------- Theme (dark / light) ---------------- */
  const THEME_KEY = "creditsense_theme";
  const root = document.documentElement;

  function applyTheme(mode) {
    root.setAttribute("data-theme", mode);
    localStorage.setItem(THEME_KEY, mode);
    document.querySelectorAll(".theme-toggle-btn i").forEach(i => {
      i.className = mode === "dark" ? "fa-solid fa-moon" : "fa-solid fa-sun";
    });
  }

  const savedTheme = localStorage.getItem(THEME_KEY) || "dark";
  applyTheme(savedTheme);

  document.querySelectorAll(".theme-toggle-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const current = root.getAttribute("data-theme") || "dark";
      applyTheme(current === "dark" ? "light" : "dark");
    });
  });

  /* ---------------- Scroll reveal ---------------- */
  const revealEls = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && revealEls.length) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    revealEls.forEach(el => io.observe(el));
  } else {
    revealEls.forEach(el => el.classList.add("in-view"));
  }

  /* ---------------- Number counters ---------------- */
  function animateCounter(el) {
    const target = parseFloat(el.dataset.count);
    const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 0;
    const suffix = el.dataset.suffix || "";
    const duration = 1600;
    const start = performance.now();
    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = target * eased;
      el.textContent = value.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ",") + suffix;
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }
  const counterEls = document.querySelectorAll("[data-count]");
  if ("IntersectionObserver" in window && counterEls.length) {
    const cio = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          cio.unobserve(entry.target);
        }
      });
    }, { threshold: 0.4 });
    counterEls.forEach(el => cio.observe(el));
  } else {
    counterEls.forEach(animateCounter);
  }

  /* ---------------- Toasts ---------------- */
  window.showToast = function (message, type = "info", icon = "fa-circle-info") {
    const stack = document.getElementById("toast-stack");
    if (!stack) return;
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.innerHTML = `<i class="fa-solid ${icon}"></i><span>${message}</span>`;
    stack.appendChild(el);
    setTimeout(() => {
      el.style.transition = "opacity .4s, transform .4s";
      el.style.opacity = "0";
      el.style.transform = "translateX(40px)";
      setTimeout(() => el.remove(), 400);
    }, 4200);
  };

  document.querySelectorAll("[data-flash]").forEach(el => {
    window.showToast(el.dataset.flash, el.dataset.flashType || "info",
      el.dataset.flashType === "success" ? "fa-circle-check" : "fa-circle-info");
  });

  /* ---------------- FAQ accordions ---------------- */
  document.querySelectorAll(".faq-q").forEach(q => {
    q.addEventListener("click", () => {
      const item = q.closest(".faq-item");
      const wasOpen = item.classList.contains("open");
      item.parentElement.querySelectorAll(".faq-item").forEach(i => i.classList.remove("open"));
      if (!wasOpen) item.classList.add("open");
    });
  });

  /* ---------------- API docs accordions ---------------- */
  document.querySelectorAll(".api-endpoint-head").forEach(head => {
    head.addEventListener("click", () => {
      head.closest(".api-endpoint").classList.toggle("open");
    });
  });

  /* ---------------- Model tabs (ML Models page) ---------------- */
  document.querySelectorAll(".model-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.target;
      document.querySelectorAll(".model-tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".model-panel").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(target).classList.add("active");
    });
  });

})();
