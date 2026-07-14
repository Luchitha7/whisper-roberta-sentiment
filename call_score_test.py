import re

from transformers import pipeline

classifier = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None,
)

CONFIDENCE_THRESHOLD = 0.6

# Placeholder weights — replace once the real list + weights come from the data analyst.
KEYWORD_WEIGHTS = {
    "cancel": -0.7,
    "complaint": -0.6,
    "refund": -0.5,
    "angry": -0.5,
    "frustrated": -0.4,
    "upset": -0.4,
    "manager": -0.3,
    "urgent": -0.2,
    "sad": -0.2,
    "happy": 0.5,
}
KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in KEYWORD_WEIGHTS) + r")\b",
    re.IGNORECASE,
)


def sentence_score(text):
    scores = {row["label"]: row["score"] for row in classifier(text)[0]}
    if max(scores.values()) < CONFIDENCE_THRESHOLD:
        return 0.0
    return scores["positive"] - scores["negative"]


def call_score(sentences):
    filtered_scores = [sentence_score(text) for text in sentences]
    avg_sentiment = sum(filtered_scores) / len(filtered_scores)

    keyword_impact = 0.0
    for text in sentences:
        for match in KEYWORD_PATTERN.findall(text):
            keyword_impact += KEYWORD_WEIGHTS[match.lower()]

    combined = avg_sentiment + keyword_impact
    clipped = max(-1.0, min(1.0, combined))
    return clipped * 100, avg_sentiment, keyword_impact


# A fake call transcript, sentence by sentence
call = [
    "Hi, thanks for calling, how can I help?",
    "I want to cancel my subscription, this is really frustrating.",
    "I've asked for a refund twice now and nobody has helped me.",
    "Can I speak to a manager please?",
    "Okay, I understand, thank you for sorting that out.",
]

score, avg_sentiment, keyword_impact = call_score(call)
print(f"Average sentiment (filtered): {avg_sentiment:.2f}")
print(f"Keyword impact: {keyword_impact:.2f}")
print(f"Overall call score: {score:.1f}%")
