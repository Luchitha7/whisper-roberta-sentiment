import threading
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from RealtimeSTT import AudioToTextRecorder
from transformers import pipeline

app = FastAPI()

MAX_HISTORY = 50
history = []
history_lock = threading.Lock()


def recorder_loop():
    classifier = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )
    recorder = AudioToTextRecorder(model="tiny", device="cpu", compute_type="int8")

    def process_text(text):
        result = classifier(text)[0]
        entry = {
            "text": text,
            "sentiment": result["label"],
            "score": round(result["score"], 2),
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        with history_lock:
            history.insert(0, entry)
            del history[MAX_HISTORY:]
        print(f'[LIVE TEXT]: "{text}" -> [SENTIMENT]: {result["label"].upper()} ({result["score"]:.2f})')

    while True:
        recorder.text(process_text)


@app.get("/history")
def get_history():
    with history_lock:
        return list(history)


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
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Live Call Sentiment</h1>
        <p class="subtitle">Speak into your microphone — updates automatically</p>
        <div id="feed"><div class="empty">Waiting for speech...</div></div>
      </div>
      <script>
        async function poll() {
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
                <div class="text">${entry.text}</div>
              </div>
            `;
          }).join('');
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
