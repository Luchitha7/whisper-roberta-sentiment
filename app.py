import threading

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from RealtimeSTT import AudioToTextRecorder
from transformers import pipeline

app = FastAPI()

latest = {"text": "", "sentiment": None, "score": None}
latest_lock = threading.Lock()


def recorder_loop():
    classifier = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )
    recorder = AudioToTextRecorder(model="tiny", device="cpu", compute_type="int8")

    def process_text(text):
        result = classifier(text)[0]
        with latest_lock:
            latest["text"] = text
            latest["sentiment"] = result["label"]
            latest["score"] = round(result["score"], 2)
        print(f'[LIVE TEXT]: "{text}" -> [SENTIMENT]: {result["label"].upper()} ({result["score"]:.2f})')

    while True:
        recorder.text(process_text)


@app.get("/latest")
def get_latest():
    with latest_lock:
        return dict(latest)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head><title>Live Call Sentiment</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 40px auto;">
      <h2>Live Call Sentiment</h2>
      <p id="text">Waiting for speech...</p>
      <p>Sentiment: <span id="sentiment">-</span> (<span id="score">-</span>)</p>
      <script>
        async function poll() {
          const res = await fetch('/latest');
          const data = await res.json();
          if (data.text) {
            document.getElementById('text').innerText = data.text;
            document.getElementById('sentiment').innerText = data.sentiment;
            document.getElementById('score').innerText = data.score;
          }
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
