from transformers import pipeline

classifier = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
)

samples = [
    "This is amazing, I love it!",
    "This is the worst experience I've ever had.",
    "The meeting is scheduled for 3pm.",
]

for text in samples:
    result = classifier(text)[0]
    print(f'"{text}" -> {result["label"]} ({result["score"]:.2f})')
