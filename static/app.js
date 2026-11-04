/* ─── MirageML — app.js ─────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
    const codeInput = document.getElementById("codeInput");
    const reviewBtn = document.getElementById("reviewBtn");
    const loadExample = document.getElementById("loadExample");
    const fileInput = document.getElementById("fileInput");
    const loadingText = document.getElementById("loadingText");
    const results = document.getElementById("results");
    const scoreBadge = document.getElementById("scoreBadge");
    const scoreValue = document.getElementById("scoreValue");
    const scoreGrade = document.getElementById("scoreGrade");
    const scoreSummary = document.getElementById("scoreSummary");
    const llmBadge = document.getElementById("llmBadge");
    const findingsContainer = document.getElementById("findingsContainer");
    const proveBtn = document.getElementById("proveBtn");
    const proveResults = document.getElementById("proveResults");

    // ─── Load Example ───
    loadExample.addEventListener("click", async () => {
        try {
            const res = await fetch("/api/example");
            const data = await res.json();
            codeInput.value = data.code;
        } catch (e) {
            codeInput.value = "# Error loading example";
        }
    });

    // ─── File Upload ───
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            codeInput.value = ev.target.result;
            // Store filename for the review
            codeInput.dataset.filename = file.name;
        };
        reader.readAsText(file);
    });

    // ─── Review Code ───
    reviewBtn.addEventListener("click", async () => {
        const code = codeInput.value.trim();
        if (!code) {
            alert("Please paste some code or load the example first.");
            return;
        }

        reviewBtn.disabled = true;
        loadingText.hidden = false;
        results.hidden = true;

        try {
            const filename = codeInput.dataset.filename || "<paste>";
            const res = await fetch("/api/review", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code, filename }),
            });

            const data = await res.json();

            if (data.error && !data.findings) {
                alert("Error: " + data.error);
                return;
            }

            renderResults(data);
        } catch (e) {
            alert("Request failed: " + e.message);
        } finally {
            reviewBtn.disabled = false;
            loadingText.hidden = true;
        }
    });

    // ─── Render Results ───
    function renderResults(data) {
        results.hidden = false;

        // Score badge
        scoreValue.textContent = data.score ?? "--";
        scoreGrade.textContent = data.grade ?? "-";
        scoreSummary.textContent = data.summary || "";

        // Grade color
        scoreBadge.className = "score-badge";
        if (data.grade) {
            scoreBadge.classList.add("grade-" + data.grade.toLowerCase());
        }

        // LLM badge
        if (data.meta && data.meta.llm_used) {
            llmBadge.textContent = "✨ Enhanced with Gemini AI explanations";
            llmBadge.hidden = false;
        } else {
            llmBadge.textContent = "📐 Rule-based analysis (no API key configured)";
            llmBadge.hidden = false;
        }

        // Findings
        findingsContainer.innerHTML = "";

        if (!data.findings || data.findings.length === 0) {
            findingsContainer.innerHTML = '<div class="no-findings">✅ No issues detected — your pipeline looks clean!</div>';
            return;
        }

        data.findings.forEach((f, i) => {
            const card = document.createElement("div");
            card.className = `finding-card severity-${f.severity}`;

            const cellInfo = f.cell ? ` · Cell ${f.cell}` : "";
            const explanation = f.explanation
                ? `<div class="finding-explanation"><strong>Impact:</strong> ${escapeHtml(f.impact || "")}<br>${escapeHtml(f.explanation)}</div>`
                : "";

            const fixCode = f.fixed_code || f.fix_hint || "";
            const fixSection = fixCode
                ? `<div class="finding-fix">
                     <h4>✅ Suggested Fix</h4>
                     <button class="copy-btn" onclick="copyFix(this)">Copy</button>
                     <pre class="fix-code">${escapeHtml(fixCode)}</pre>
                   </div>`
                : "";

            card.innerHTML = `
                <div class="finding-header">
                    <span class="finding-title">${escapeHtml(f.title)}</span>
                    <div class="finding-meta">
                        <span class="severity-tag ${f.severity}">${f.severity}</span>
                        <span class="rule-tag">${f.rule_id}</span>
                        <span class="line-tag">Line ${f.line}${cellInfo}</span>
                    </div>
                </div>
                <pre class="finding-snippet">${escapeHtml(f.snippet || "")}</pre>
                <div class="finding-why">${escapeHtml(f.why || "")}</div>
                ${explanation}
                ${fixSection}
            `;

            findingsContainer.appendChild(card);
        });

        // Scroll to results
        results.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // ─── Prove It ───
    proveBtn.addEventListener("click", async () => {
        proveBtn.disabled = true;
        proveBtn.textContent = "Running...";

        try {
            const res = await fetch("/api/prove", { method: "POST" });
            const data = await res.json();

            if (data.error) {
                alert("Error: " + data.error);
                return;
            }

            proveResults.hidden = false;

            // Animate bars
            const leakyBar = document.getElementById("leakyBar");
            const correctBar = document.getElementById("correctBar");
            const leakyScore = document.getElementById("leakyScore");
            const correctScore = document.getElementById("correctScore");

            setTimeout(() => {
                leakyBar.style.width = (data.leaky_score * 100) + "%";
                leakyBar.textContent = (data.leaky_score * 100).toFixed(1) + "%";
                correctBar.style.width = (data.correct_score * 100) + "%";
                correctBar.textContent = (data.correct_score * 100).toFixed(1) + "%";
            }, 100);

            leakyScore.textContent = `Leaky: ${(data.leaky_score * 100).toFixed(1)}%`;
            correctScore.textContent = `Correct: ${(data.correct_score * 100).toFixed(1)}%`;

            document.getElementById("proveDescription").textContent = data.description;
            document.getElementById("proveMethod").textContent = data.method;
        } catch (e) {
            alert("Request failed: " + e.message);
        } finally {
            proveBtn.disabled = false;
            proveBtn.textContent = "Run Experiment";
        }
    });

    // ─── Helpers ───
    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }
});

// Global copy function
function copyFix(btn) {
    const code = btn.nextElementSibling.textContent;
    navigator.clipboard.writeText(code).then(() => {
        btn.textContent = "Copied!";
        setTimeout(() => { btn.textContent = "Copy"; }, 2000);
    });
}
