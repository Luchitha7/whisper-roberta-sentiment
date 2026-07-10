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
KEYWORDS = [
    "refund", "cancel", "manager", "complaint", "urgent",
    "happy", "sad", "angry", "frustrated", "upset",
]
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
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
      <style>
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
          background-color: #f3f2ef;
          background-image: radial-gradient(circle at 12% 8%, #ffffff 0%, rgba(255,255,255,0) 45%),
                             radial-gradient(circle at 92% 90%, #eae7e0 0%, rgba(234,231,224,0) 50%);
          min-height: 100vh;
          color: #23252b;
        }
        .page {
          min-height: 100vh;
          padding: 56px 24px;
          box-sizing: border-box;
          display: flex;
          justify-content: center;
        }
        .container { width: 100%; max-width: 760px; }
        .title-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
        .live-dot {
          width: 10px; height: 10px; border-radius: 50%;
          background: #ef4444; display: inline-block;
          animation: pulse 1.6s ease-in-out infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
        h1 {
          font-size: 28px;
          font-weight: 800;
          color: #14151a;
          margin: 0;
          letter-spacing: -0.02em;
        }
        .subtitle {
          color: #70747e;
          font-size: 15px;
          margin: 0 0 28px;
        }
        .keyword-panel {
          background: #ffffff;
          border: 1px solid #e6e7eb;
          border-radius: 16px;
          padding: 24px 28px;
          margin-bottom: 20px;
          box-shadow: 0 1px 2px rgba(20,21,26,0.04);
        }
        .keyword-panel h2 {
          font-size: 12px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: #9a9ea8;
          margin: 0 0 14px;
        }
        .chips { display: flex; flex-wrap: wrap; gap: 10px; }
        .chip {
          font-size: 14px;
          font-weight: 600;
          padding: 7px 14px;
          border-radius: 999px;
          background: #f0f1f4;
          color: #5c606a;
        }
        .chip.hit {
          background: #fdeaea;
          color: #c62828;
        }
        .empty {
          color: #9a9ea8;
          text-align: center;
          padding: 40px 0;
          font-size: 14px;
        }
        #feed { display: flex; flex-direction: column; gap: 14px; }
        .card {
          background: #ffffff;
          border: 1px solid #ececec;
          border-radius: 12px;
          padding: 18px 22px;
          box-shadow: 0 1px 2px rgba(20,21,26,0.04);
        }
        .card-top {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .badge {
          font-size: 12px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          padding: 4px 10px;
          border-radius: 6px;
        }
        .badge.positive { background: #22a06b; color: #ffffff; }
        .badge.negative { background: #dc4444; color: #ffffff; }
        .badge.neutral { background: #eceef1; color: #5c606a; }
        .time { font-size: 13px; color: #a2a6ae; font-variant-numeric: tabular-nums; }
        .text { font-size: 16px; line-height: 1.5; color: #23252b; }
        .text .keyword { color: #c62828; font-weight: 700; }
      </style>
    </head>
    <body>
      <div class="page">
      <div class="container">
        <div class="title-row">
          <span class="live-dot"></span>
          <h1>Live Call Sentiment</h1>
        </div>
        <p class="subtitle">Speak into your microphone — updates automatically</p>
        <div class="keyword-panel">
          <h2>Keyword count</h2>
          <div class="chips" id="chips"></div>
        </div>
        <div id="feed"><div class="empty">Waiting for speech...</div></div>
      </div>
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
              <div class="card">
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
