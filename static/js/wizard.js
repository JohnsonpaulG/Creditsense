/* =====================================================================
   Prediction Wizard — 5 step form controller
   ===================================================================== */
(function () {
  "use strict";
  const form = document.getElementById("wizard-form");
  if (!form) return;

  const panels = Array.from(document.querySelectorAll(".wizard-panel"));
  const steps = Array.from(document.querySelectorAll(".w-step"));
  const progressLine = document.querySelector(".wizard-steps .progress-line");
  const btnBackEls = document.querySelectorAll("[data-wizard-back]");
  const btnNextEls = document.querySelectorAll("[data-wizard-next]");
  let current = 0;

  function requiredFieldsFor(panel) {
    return Array.from(panel.querySelectorAll("input[required], select[required]"));
  }

  function validatePanel(panel) {
    let valid = true;
    requiredFieldsFor(panel).forEach(field => {
      const wrap = field.closest(".field");
      const empty = !field.value || field.value.trim() === "";
      const numInvalid = field.type === "number" && field.value !== "" &&
        ((field.min !== "" && Number(field.value) < Number(field.min)) ||
         (field.max !== "" && Number(field.value) > Number(field.max)));
      if (empty || numInvalid) {
        valid = false;
        wrap && wrap.classList.add("error");
      } else {
        wrap && wrap.classList.remove("error");
      }
    });
    return valid;
  }

  function showPanel(index) {
    panels.forEach((p, i) => p.classList.toggle("active", i === index));
    steps.forEach((s, i) => {
      s.classList.toggle("active", i === index);
      s.classList.toggle("done", i < index);
    });
    if (progressLine) {
      const pct = (index / (steps.length - 1)) * 100;
      progressLine.style.width = pct + "%";
    }
    window.scrollTo({ top: document.querySelector(".wizard-wrap").offsetTop - 90, behavior: "smooth" });
    if (index === panels.length - 1) buildReview();
  }

  btnNextEls.forEach(btn => {
    btn.addEventListener("click", () => {
      const panel = panels[current];
      if (!validatePanel(panel)) {
        window.showToast && window.showToast("Please complete all required fields.", "danger", "fa-triangle-exclamation");
        return;
      }
      if (current < panels.length - 1) {
        current++;
        showPanel(current);
      }
    });
  });

  btnBackEls.forEach(btn => {
    btn.addEventListener("click", () => {
      if (current > 0) {
        current--;
        showPanel(current);
      }
    });
  });

  /* ---------------- choice-card style radio replacements ---------------- */
  document.querySelectorAll(".choice-card").forEach(card => {
    card.addEventListener("click", () => {
      const groupName = card.dataset.group;
      const hiddenInput = form.querySelector(`input[name="${groupName}"]`);
      document.querySelectorAll(`.choice-card[data-group="${groupName}"]`).forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      if (hiddenInput) hiddenInput.value = card.dataset.value;
    });
  });

  /* ---------------- Build review panel (step 5) ---------------- */
  function buildReview() {
    const reviewGrid = document.getElementById("review-grid");
    if (!reviewGrid) return;
    const data = new FormData(form);
    const labels = {
      applicant_name: "Applicant Name", gender: "Gender", age: "Age",
      own_car: "Owns Car", own_house: "Owns Property", children: "Children",
      income: "Annual Income", income_type: "Income Type", education: "Education",
      family_status: "Family Status", housing_type: "Housing Type",
      occupation: "Occupation", family_members: "Family Members",
      years_employed: "Years Employed",
    };
    let html = "";
    Object.keys(labels).forEach(key => {
      let val = data.get(key) || "—";
      if (key === "income" && val !== "—") val = "$" + Number(val).toLocaleString();
      html += `<div class="review-item"><span class="k">${labels[key]}</span><span class="v">${val}</span></div>`;
    });
    reviewGrid.innerHTML = html;
  }

  /* ---------------- Submit -> loading overlay ---------------- */
  form.addEventListener("submit", () => {
    const overlay = document.getElementById("predict-loading");
    if (!overlay) return;
    overlay.classList.add("show");
    const msgEl = overlay.querySelector(".step-msgs");
    const msgs = [
      "Validating applicant details…",
      "Encoding categorical features…",
      "Scaling feature vector…",
      "Running Random Forest inference…",
      "Calculating risk & credit limit…",
    ];
    let i = 0;
    if (msgEl) {
      msgEl.textContent = msgs[0];
      const interval = setInterval(() => {
        i = (i + 1) % msgs.length;
        msgEl.textContent = msgs[i];
      }, 550);
      window.addEventListener("beforeunload", () => clearInterval(interval));
    }
  });

  showPanel(0);
})();
