from dataclasses import dataclass

@dataclass(frozen=True)
class HFData:
    unsafe_dataset: str = "LocalDoc/pii_ner_azerbaijani"
    unsafe_text_col: str = "translated_text"

    safe_dataset: str = "aznlp/azerbaijani-blogs"
    safe_text_col: str = "content"

@dataclass(frozen=True)
class Models:
    classifier_base: str = "distilbert-base-multilingual-cased"
    ner_base: str = "distilbert-base-multilingual-cased"

DATA = HFData()
MODELS = Models()