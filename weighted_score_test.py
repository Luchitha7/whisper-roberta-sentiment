from transformers import pipeline

classifier = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None,
)

CONFIDENCE_THRESHOLD = 0.6


def sentence_score(text):
    scores = {row["label"]: row["score"] for row in classifier(text)[0]}
    positive = scores["positive"]
    negative = scores["negative"]

    highest = max(scores.values())
    if highest < CONFIDENCE_THRESHOLD:
        return 0.0, scores

    return positive - negative, scores


samples = [
    "This is amazing, I love it!",
    "This is the worst experience I've ever had.",
    "The meeting is scheduled for 3pm.",
    "I guess it's fine, not sure really.",
]

for text in samples:
    score, scores = sentence_score(text)
    breakdown = ", ".join(f'{label}: {value:.2f}' for label, value in scores.items())
    print(f'"{text}"')
    print(f'  scores -> {breakdown}')
    print(f'  Sraw/filtered -> {score:.2f}')
    print()
