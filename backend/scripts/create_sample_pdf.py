"""
Create a small multi-page sample PDF for trying out the API.

    python scripts/create_sample_pdf.py

Writes samples/sample_nlp_document.pdf
"""

from pathlib import Path

import pymupdf

PAGES = [
    (
        "Introduction to Natural Language Processing",
        "Natural Language Processing (NLP) is a branch of artificial intelligence that enables "
        "computers to understand, interpret and generate human language. Common NLP tasks include "
        "text classification, named entity recognition, machine translation and question answering. "
        "Early NLP systems were based on hand-written rules, while modern systems learn from data.",
    ),
    (
        "Text Representation",
        "Before text can be processed by a machine learning model it must be converted into numbers. "
        "TF-IDF (Term Frequency - Inverse Document Frequency) is a classical representation that "
        "weights a word by how often it appears in a document and how rare it is across documents. "
        "Cosine similarity between TF-IDF vectors is widely used to measure how similar two texts are.",
    ),
    (
        "The Transformer Architecture",
        "The Transformer architecture was introduced in 2017 by Vaswani and colleagues in the paper "
        "Attention Is All You Need. Instead of recurrence, it relies on a mechanism called "
        "self-attention, which lets every token look at every other token in the sequence. BERT, "
        "released by Google in 2018, is an encoder-only Transformer pretrained on large text corpora.",
    ),
    (
        "DistilBERT and Question Answering",
        "DistilBERT is a smaller version of BERT created by Hugging Face using knowledge distillation. "
        "It has 40 percent fewer parameters than BERT and runs 60 percent faster, while retaining "
        "about 97 percent of BERT's language understanding performance. The model "
        "distilbert-base-cased-distilled-squad was fine-tuned on the Stanford Question Answering "
        "Dataset (SQuAD), which contains more than 100,000 questions. Given a question and a passage, "
        "it predicts the start and end positions of the answer span inside the passage.",
    ),
]


def main() -> None:
    output = Path(__file__).resolve().parent.parent / "samples" / "sample_nlp_document.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open()
    for title, body in PAGES:
        page = document.new_page()
        page.insert_text((72, 80), title, fontsize=16)
        rect = pymupdf.Rect(72, 110, page.rect.width - 72, page.rect.height - 72)
        page.insert_textbox(rect, body, fontsize=11)
    document.save(output)
    document.close()
    print(f"Sample PDF written to {output}")


if __name__ == "__main__":
    main()

