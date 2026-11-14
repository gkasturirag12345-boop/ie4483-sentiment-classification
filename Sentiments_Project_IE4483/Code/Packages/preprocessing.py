"""Text normalisation and tokenisation shared by the TF-IDF and BiLSTM models.

Design choices (keep sentiment cues):
- lowercase, HTML-unescape, collapse whitespace/newlines;
- NO stop-word removal, so negations ("not", "never", "no") survive;
- keep apostrophes inside words ("don't", "isn't") instead of sklearn's default
  pattern, which splits "don't" into "don" + a dropped "t";
- keep "!" and "?" as separate tokens (emphasis carries sentiment).
DistilBERT does not use this module: it uses its own WordPiece tokenizer on raw text.
"""
import html
import re

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]")
TOKEN_PATTERN = r"(?u)[a-z0-9]+(?:'[a-z]+)?|[!?]"  # same pattern, for TfidfVectorizer


def normalise(text: str) -> str:
    text = html.unescape(str(text)).lower()
    text = text.replace("’", "'").replace("`", "'")  # curly/back quotes -> apostrophe
    return _WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalise(text))


def group_key(text: str) -> str:
    """Key used to detect duplicates / near-duplicates (case, whitespace and punctuation-insensitive)."""
    return " ".join(re.findall(r"[a-z0-9']+", normalise(text)))


if __name__ == "__main__":
    for s in ["I don't think it's GOOD!!", "Never works.\n\nNot worth &amp; the money?"]:
        print(s, "->", tokenize(s))
