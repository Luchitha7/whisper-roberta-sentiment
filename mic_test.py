from RealtimeSTT import AudioToTextRecorder


def process_text(text):
    print(f"You said: {text}")


if __name__ == '__main__':
    print("Initializing... say something once it's ready. Press Ctrl+C to stop.")
    recorder = AudioToTextRecorder(model="tiny", device="cpu", compute_type="int8")
    while True:
        recorder.text(process_text)
