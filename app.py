import html
import re
import threading
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from RealtimeSTT import AudioToTextRecorder
from transformers import pipeline

app = FastAPI()

# Placeholder list — replace with the real words from the data analyst.
KEYWORDS = ["refund", "cancel", "manager", "complaint", "urgent"]
KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in KEYWORDS) + r")\b",
    re.IGNORECASE,
)

MAX_HISTORY = 50
history = []
keyword_counts = {word: 0 for word in KEYWORDS}
state_lock = threading.Lock()


def highlight_keywords(text):
    escaped = html.escape(text)
    return KEYWORD_PATTERN.sub(r'<span class="keyword">\1</span>', escaped)


def recorder_loop():
    classifier = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )
    recorder = AudioToTextRecorder(model="tiny", device="cpu", compute_type="int8")

    def process_text(text):
        result = classifier(text)[0]
        matches = KEYWORD_PATTERN.findall(text)

        entry = {
            "text": text,
            "highlighted": highlight_keywords(text),
            "sentiment": result["label"],
            "score": round(result["score"], 2),
            "time": datetime.now().strftime("%H:%M:%S"),
        }

        with state_lock:
            history.insert(0, entry)
            del history[MAX_HISTORY:]
            for match in matches:
                keyword_counts[match.lower()] += 1

        print(f'[LIVE TEXT]: "{text}" -> [SENTIMENT]: {result["label"].upper()} ({result["score"]:.2f})')
        if matches:
            print(f'[KEYWORDS]: {", ".join(matches)}')

    while True:
        recorder.text(process_text)


@app.get("/history")
def get_history():
    with state_lock:
        return list(history)


@app.get("/keywords")
def get_keywords():
    with state_lock:
        return dict(keyword_counts)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head>
      <title>Live Call Sentiment</title>
      <style>
        :root {
          --positive: #2e7d32;
          --negative: #c62828;
          --neutral: #616161;
          --bg: #f4f5f7;
          --card-bg: #ffffff;
        }
        * { box-sizing: border-box; }
        body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: var(--bg);
          margin: 0;
          padding: 24px;
          color: #1c1c1e;
        }
        .container { max-width: 640px; margin: 0 auto; }
        h1 {
          font-size: 20px;
          font-weight: 600;
          margin: 0 0 4px;
        }
        .subtitle {
          color: #6e6e73;
          font-size: 13px;
          margin: 0 0 20px;
        }
        .keyword-panel {
          background: var(--card-bg);
          border-radius: 8px;
          padding: 12px 16px;
          margin-bottom: 20px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        }
        .keyword-panel h2 {
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: #6e6e73;
          margin: 0 0 8px;
        }
        .chips { display: flex; flex-wrap: wrap; gap: 8px; }
        .chip {
          font-size: 13px;
          padding: 4px 10px;
          border-radius: 12px;
          background: #f4f5f7;
          color: #1c1c1e;
        }
        .chip.hit {
          background: #ffebee;
          color: var(--negative);
          font-weight: 600;
        }
        .empty {
          color: #8e8e93;
          text-align: center;
          padding: 40px 0;
          font-size: 14px;
        }
        .card {
          background: var(--card-bg);
          border-left: 4px solid var(--neutral);
          border-radius: 8px;
          padding: 12px 16px;
          margin-bottom: 10px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        }
        .card.positive { border-left-color: var(--positive); }
        .card.negative { border-left-color: var(--negative); }
        .card.neutral { border-left-color: var(--neutral); }
        .card-top {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 6px;
        }
        .badge {
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.03em;
          padding: 2px 8px;
          border-radius: 10px;
          color: white;
        }
        .badge.positive { background: var(--positive); }
        .badge.negative { background: var(--negative); }
        .badge.neutral { background: var(--neutral); }
        .time { font-size: 12px; color: #8e8e93; }
        .text { font-size: 15px; line-height: 1.4; }
        .text .keyword {
          background: #ffebee;
          color: var(--negative);
          font-weight: 600;
          padding: 0 3px;
          border-radius: 3px;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Live Call Sentiment</h1>
        <p class="subtitle">Speak into your microphone — updates automatically</p>
        <div class="keyword-panel">
          <h2>Keyword count</h2>
          <div class="chips" id="chips"></div>
        </div>
        <div id="feed"><div class="empty">Waiting for speech...</div></div>
      </div>
      <script>
        async function pollHistory() {
          const res = await fetch('/history');
          const data = await res.json();
          const feed = document.getElementById('feed');
          if (data.length === 0) {
            feed.innerHTML = '<div class="empty">Waiting for speech...</div>';
            return;
          }
          feed.innerHTML = data.map(entry => {
            const sentiment = (entry.sentiment || 'neutral').toLowerCase();
            const pct = Math.round(entry.score * 100);
            return `
              <div class="card ${sentiment}">
                <div class="card-top">
                  <span class="badge ${sentiment}">${sentiment} · ${pct}%</span>
                  <span class="time">${entry.time}</span>
                </div>
                <div class="text">${entry.highlighted}</div>
              </div>
            `;
          }).join('');
        }

        async function pollKeywords() {
          const res = await fetch('/keywords');
          const data = await res.json();
          const chips = document.getElementById('chips');
          chips.innerHTML = Object.entries(data).map(([word, count]) => {
            const hit = count > 0 ? 'hit' : '';
            return `<span class="chip ${hit}">${word}: ${count}</span>`;
          }).join('');
        }

        function poll() {
          pollHistory();
          pollKeywords();
        }

        setInterval(poll, 1000);
        poll();
      </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    threading.Thread(target=recorder_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
