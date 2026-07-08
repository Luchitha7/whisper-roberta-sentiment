from RealtimeSTT import AudioToTextRecorder
from transformers import pipeline

classifier = None


def process_text(text):
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]
    print(f'[LIVE TEXT]: "{text}" -> [SENTIMENT]: {label.upper()} ({score:.2f})')


if __name__ == '__main__':
    print("Loading sentiment model...")
    classifier = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )

    print("Initializing recorder... say something once it's ready. Press Ctrl+C to stop.")
    recorder = AudioToTextRecorder(model="tiny", device="cpu", compute_type="int8")
    while True:
        recorder.text(process_text)
