/* =====================================================================
   History page — search / filter / sort / paginate (client-side)
   ===================================================================== */
(function () {
  "use strict";
  const table = document.getElementById("history-table");
  if (!table) return;

  const tbody = table.querySelector("tbody");
  const allRows = Array.from(tbody.querySelectorAll("tr"));
  const searchInput = document.getElementById("history-search");
  const filterChips = document.querySelectorAll("#history-filters .filter-chip");
  const pagination = document.getElementById("history-pagination");
  const PAGE_SIZE = 10;

  let currentFilter = "all";
  let currentSearch = "";
  let currentPage = 1;
  let sortKey = null;
  let sortAsc = true;

  function matches(row) {
    const name = row.dataset.name || "";
    const status = row.dataset.status || "";
    const risk = row.dataset.risk || "";
    if (currentSearch && !name.includes(currentSearch)) return false;
    if (currentFilter === "approved" && status !== "approved") return false;
    if (currentFilter === "declined" && status !== "declined") return false;
    if (currentFilter === "low" && risk !== "low-risk") return false;
    if (currentFilter === "high" && risk !== "high-risk") return false;
    return true;
  }

  function render() {
    const filtered = allRows.filter(matches);
    const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
    currentPage = Math.min(currentPage, totalPages);

    allRows.forEach(r => r.style.display = "none");
    const start = (currentPage - 1) * PAGE_SIZE;
    filtered.slice(start, start + PAGE_SIZE).forEach(r => r.style.display = "");

    pagination.innerHTML = "";
    for (let i = 1; i <= totalPages; i++) {
      const btn = document.createElement("button");
      btn.textContent = i;
      if (i === currentPage) btn.classList.add("active");
      btn.addEventListener("click", () => { currentPage = i; render(); });
      pagination.appendChild(btn);
    }

    // toggle empty message
    let emptyRow = tbody.querySelector(".no-results-row");
    if (filtered.length === 0) {
      if (!emptyRow) {
        emptyRow = document.createElement("tr");
        emptyRow.className = "no-results-row";
        emptyRow.innerHTML = `<td colspan="7" style="text-align:center;padding:40px;color:var(--text-2)">No matching predictions found.</td>`;
        tbody.appendChild(emptyRow);
      }
      emptyRow.style.display = "";
    } else if (emptyRow) {
      emptyRow.style.display = "none";
    }
  }

  if (searchInput) {
    searchInput.addEventListener("input", () => {
      currentSearch = searchInput.value.trim().toLowerCase();
      currentPage = 1;
      render();
    });
  }

  filterChips.forEach(chip => {
    chip.addEventListener("click", () => {
      filterChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      currentFilter = chip.dataset.filter;
      currentPage = 1;
      render();
    });
  });

  table.querySelectorAll("th[data-sort]").forEach((th, colIndex) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      sortAsc = sortKey === key ? !sortAsc : true;
      sortKey = key;
      allRows.sort((a, b) => {
        const ca = a.children[colIndex].textContent.trim();
        const cb = b.children[colIndex].textContent.trim();
        const na = parseFloat(ca.replace(/[^0-9.\-]/g, ""));
        const nb = parseFloat(cb.replace(/[^0-9.\-]/g, ""));
        let cmp;
        if (!isNaN(na) && !isNaN(nb) && /^[#$0-9.,%\s]+$/.test(ca)) {
          cmp = na - nb;
        } else {
          cmp = ca.localeCompare(cb);
        }
        return sortAsc ? cmp : -cmp;
      });
      allRows.forEach(r => tbody.appendChild(r));
      render();
    });
  });

  render();
})();
