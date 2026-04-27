import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pipeline


HOST = "127.0.0.1"
PORT = 8000


PIPELINE = None


def build_analysis_context(category: str, visibility: str) -> str:
    normalized_category = category.strip() or "SNS"
    normalized_visibility = visibility.strip().lower()
    visibility_label = "Private" if normalized_visibility == "private" else "Public"
    return f"{normalized_category} / {visibility_label}"


def load_pipeline() -> dict:
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    PIPELINE = pipeline.__dict__
    return PIPELINE


def result_to_payload(result) -> dict:
    return {
        "input": result.original_text,
        "normalized": result.normalized_text,
        "context": result.context,
        "riskScore": result.final_risk_score,
        "riskLevel": result.risk_level,
        "thresholdType": result.threshold_type,
        "categories": result.issue_categories,
        "recommended": result.recommended_context,
        "rewrite": result.rewrite_suggestion,
        "llmBackend": result.llm_backend,
        "problematicPhrases": result.problematic_phrases,
        "specialSituations": result.special_situations,
        "explanations": result.explanation_points,
        "dimensions": [
            {
                "name": name,
                "heuristic": round(float(result.heuristic_dimensions[name]), 3),
                "final": round(float(result.final_dimensions[name]), 3),
                "llmProbability": round(float(result.llm_dimension_scores[name]["probability"]), 3),
                "llmSeverity": round(float(result.llm_dimension_scores[name]["severity"]), 3),
                "llmConfidence": round(float(result.llm_dimension_scores[name]["confidence"]), 3),
                "rubricReason": str(result.llm_dimension_scores[name].get("rubric_reason", "")),
            }
            for name in load_pipeline()["DIMENSION_ORDER"]
        ],
        "validationWarnings": result.validation_report.get("warnings", []),
    }


def run_analysis(text: str, context: str, visibility: str, backend: str = "ollama") -> dict:
    pipeline = load_pipeline()
    if backend in {"disabled", "ollama", "auto", "transformers"}:
        pipeline["QWEN_BACKEND"] = backend

    context = build_analysis_context(context, visibility)
    result = pipeline["run_backlash_risk_pipeline"](text, context=context)
    return result_to_payload(result)


def render_index() -> bytes:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Backlash Risk Analyzer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f7f6;
      --panel: #ffffff;
      --ink: #17201b;
      --muted: #64716b;
      --line: #d8e0dc;
      --accent: #0f766e;
      --accent-strong: #0b4f4a;
      --warn: #b45309;
      --danger: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "Malgun Gothic", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 28px auto;
      display: grid;
      grid-template-columns: minmax(320px, 430px) 1fr;
      gap: 18px;
      align-items: start;
    }}
    section, form {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    h1 {{
      grid-column: 1 / -1;
      margin: 0 0 2px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    label {{
      display: block;
      margin: 14px 0 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }}
    textarea, input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }}
    textarea {{
      min-height: 170px;
      resize: vertical;
      line-height: 1.45;
    }}
    button {{
      margin-top: 16px;
      width: 100%;
      border: 0;
      border-radius: 6px;
      padding: 12px 14px;
      font: inherit;
      font-weight: 700;
      color: #fff;
      background: var(--accent);
      cursor: pointer;
    }}
    button:disabled {{
      opacity: .65;
      cursor: wait;
    }}
    .result-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
      margin-bottom: 14px;
    }}
    .score {{
      font-size: 38px;
      font-weight: 800;
      color: var(--accent-strong);
      min-width: 74px;
      text-align: right;
    }}
    .muted {{ color: var(--muted); }}
    .recommended {{
      border-left: 4px solid var(--accent);
      padding: 10px 12px;
      background: #eef8f6;
      border-radius: 4px;
      margin: 12px 0;
      line-height: 1.45;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0 14px;
    }}
    .chip {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      background: #fafcfb;
      font-size: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); }}
    ul {{ padding-left: 18px; }}
    .error {{
      color: var(--danger);
      background: #fff1f2;
      border: 1px solid #fecdd3;
      padding: 12px;
      border-radius: 6px;
      white-space: pre-wrap;
    }}
    @media (max-width: 820px) {{
      main {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Backlash Risk Analyzer</h1>
    <form id="analyze-form">
      <label for="text">Input</label>
      <textarea id="text" name="text" placeholder="Input your text here..."></textarea>
      <label for="context">Category</label>
      <select id="context" name="context">
        <option value="SNS" selected>SNS</option>
        <option value="Email">Email</option>
        <option value="Message">Message</option>
      </select>
      <label for="visibility">Scope of Disclosure</label>
      <select id="visibility" name="visibility">
        <option value="public" selected>public</option>
        <option value="private">private</option>
      </select>
      <button id="submit-button" type="submit">Analyze</button>
    </form>
    <section id="output">
      <p class="muted">Analysis results will be displayed here.</p>
    </section>
  </main>
  <script>
    const form = document.getElementById("analyze-form");
    const button = document.getElementById("submit-button");
    const output = document.getElementById("output");

    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, char => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }}[char]));
    }}

    function renderList(items) {{
      if (!items || !items.length) return "<p class='muted'>None</p>";
      return `<ul>${{items.map(item => `<li>${{escapeHtml(item)}}</li>`).join("")}}</ul>`;
    }}

    function renderResult(data) {{
      const chips = (data.categories || []).map(item => `<span class="chip">${{escapeHtml(item)}}</span>`).join("");
      const rows = (data.dimensions || []).map(item => `
        <tr>
          <td>${{escapeHtml(item.name)}}</td>
          <td>${{item.heuristic}}</td>
          <td>${{item.final}}</td>
          <td>${{item.llmProbability}}</td>
          <td>${{item.llmConfidence}}</td>
        </tr>
      `).join("");

      output.innerHTML = `
        <div class="result-header">
          <div>
            <div class="muted">${{escapeHtml(data.riskLevel)}} · ${{escapeHtml(data.llmBackend)}}</div>
            <h2>${{escapeHtml(data.context)}}</h2>
          </div>
          <div class="score">${{data.riskScore}}</div>
        </div>
        <div class="chips">${{chips}}</div>
        <div class="recommended">${{escapeHtml(data.recommended)}}</div>
        <p class="muted"><strong>Normalized:</strong> ${{escapeHtml(data.normalized)}}</p>
        <h3>Problematic Phrases</h3>
        ${{renderList(data.problematicPhrases)}}
        <h3>Why</h3>
        ${{renderList(data.explanations)}}
        <h3>Dimensions</h3>
        <table>
          <thead><tr><th>Dimension</th><th>Heuristic</th><th>Final</th><th>LLM P</th><th>LLM Conf.</th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>
      `;
    }}

    form.addEventListener("submit", async event => {{
      event.preventDefault();
      button.disabled = true;
      button.textContent = "Analyzing...";
      output.innerHTML = "<p class='muted'>Analyzing...</p>";
      const body = {{
        text: form.text.value,
        context: form.context.value,
        visibility: form.visibility.value
      }};
      try {{
        const response = await fetch("/api/analyze", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(body)
        }});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Request failed");
        renderResult(data);
      }} catch (error) {{
        output.innerHTML = `<div class="error">${{escapeHtml(error.message)}}</div>`;
      }} finally {{
        button.disabled = false;
        button.textContent = "Analyze";
      }}
    }});
  </script>
</body>
</html>""".encode("utf-8")


class AppHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send(200, render_index(), "text/html; charset=utf-8")
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            body = json.loads(raw_body) if raw_body else {}
            text = str(body.get("text", "")).strip()
            context = str(body.get("context", "Instagram / Public")).strip() or "Instagram / Public"
            visibility = str(body.get("visibility", "public")).strip() or "public"

            if not text:
                raise ValueError("Input text is required.")

            payload = run_analysis(text=text, context=context, visibility=visibility)
            response = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(200, response, "application/json; charset=utf-8")
        except Exception as exc:
            payload = {
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
            }
            response = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(500, response, "application/json; charset=utf-8")

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    load_pipeline()
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Serving on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
