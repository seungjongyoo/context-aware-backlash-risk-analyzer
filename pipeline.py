"""Production pipeline for social backlash risk scoring."""

import html
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    class _SimpleVectorResult:
        def __init__(self, rows: List[List[float]]):
            self.rows = rows

        def toarray(self) -> np.ndarray:
            return np.asarray(self.rows, dtype=float)

    class TfidfVectorizer:
        def __init__(self, ngram_range: tuple = (1, 1), min_df: int = 1):
            self.vocabulary_: Dict[str, int] = {}

        def fit(self, texts: List[str]) -> "TfidfVectorizer":
            terms = sorted({term for text in texts for term in re.findall(r"\b\w+\b", text.lower())})
            self.vocabulary_ = {term: index for index, term in enumerate(terms)}
            return self

        def transform(self, texts: List[str]) -> _SimpleVectorResult:
            rows = []
            for text in texts:
                row = [0.0] * len(self.vocabulary_)
                for term in re.findall(r"\b\w+\b", text.lower()):
                    if term in self.vocabulary_:
                        row[self.vocabulary_[term]] += 1.0
                norm = math.sqrt(sum(value * value for value in row)) or 1.0
                rows.append([value / norm for value in row])
            return _SimpleVectorResult(rows)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMER_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


def torch_cuda_available() -> bool:
    return bool(torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available())


def get_runtime_device() -> str:
    return "cuda" if torch_cuda_available() else "cpu"


def get_gpu_memory_gb() -> float:
    if not torch_cuda_available():
        return 0.0
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        return round(total_bytes / (1024 ** 3), 2)
    except Exception:
        return 0.0


RUNTIME_DEVICE = get_runtime_device()
GPU_MEMORY_GB = get_gpu_memory_gb()
MIN_QWEN_TRANSFORMERS_VRAM_GB = 14.0

QWEN_BACKEND = "ollama"  # auto | ollama | transformers | disabled
QWEN_OLLAMA_MODEL = "qwen3:8b"
QWEN_OLLAMA_URL = "http://localhost:11434/api/generate"
QWEN_TRANSFORMERS_MODEL = "Qwen/Qwen3-8B"
SOCIAL_CONTEXT = "Instagram / Public"
AUDIENCE_PROFILE = (
    "Public Instagram audience reading a short opinion post. Readers include casual followers, "
    "fans of the target, neutral bystanders, and people sensitive to sarcasm, dunking, or sweeping claims."
)

# Human-designed taxonomy
DIMENSION_ORDER = [
    "Aggression",
    "Group Generalization",
    "Sarcasm / Mockery",
    "Overconfident Judgment",
    "Context Inappropriateness",
    "Misinterpretability",
    "Norm Violation",
]

DIMENSION_DESCRIPTIONS = {
    "Aggression": "Level of direct aggression or harsh criticism",
    "Group Generalization": "Degree of broad judgment about an entire group",
    "Sarcasm / Mockery": "Intensity of sarcasm, mockery, or derision",
    "Overconfident Judgment": "Degree to which a personal opinion is framed like an absolute fact",
    "Context Inappropriateness": "Likelihood of appearing inappropriate in a public SNS context",
    "Misinterpretability": "Likelihood that the intent could be distorted or read as hostile",
    "Norm Violation": "Tendency toward discriminatory, dehumanizing, or norm-violating language",
}


# Human-designed rubric
DIMENSION_RUBRICS = {
    "Aggression": {
        "probability_meaning": "How likely readers are to perceive the post as directly hostile, insulting, or confrontational.",
        "severity_meaning": "How damaging the perceived hostility would be to the speaker's reputation or discussion quality.",
        "levels": {
            "0.0-0.2": "Very low: neutral disagreement, mild dissatisfaction, or no identifiable hostile target.",
            "0.2-0.4": "Mild: some bluntness or irritation, but not clearly attacking a person, group, or target.",
            "0.4-0.6": "Moderate: clearly harsh wording or negative targeting that many readers may see as aggressive.",
            "0.6-0.8": "High: direct insult, hostile phrasing, or attack-like wording likely to draw criticism.",
            "0.8-1.0": "Very high: severe hostility, degrading language, or intense direct attack with strong backlash risk.",
        },
    },
    "Group Generalization": {
        "probability_meaning": "How likely readers are to perceive a broad claim about an entire group or category of people.",
        "severity_meaning": "How damaging the generalization would be if readers treat it as stereotyping or unfair grouping.",
        "levels": {
            "0.0-0.2": "Very low: refers to a specific case without broadening to a group.",
            "0.2-0.4": "Mild: vague plural language but limited or softened scope.",
            "0.4-0.6": "Moderate: noticeable broad framing that may overstate a pattern.",
            "0.6-0.8": "High: sweeping judgment about many people or a recognizable group.",
            "0.8-1.0": "Very high: categorical condemnation or stereotype-like claim about a group.",
        },
    },
    "Sarcasm / Mockery": {
        "probability_meaning": "How likely readers are to perceive sarcasm, ridicule, dunking, or dismissive humor.",
        "severity_meaning": "How damaging the mocking tone would be if readers notice it.",
        "levels": {
            "0.0-0.2": "Very low: sincere or neutral tone with no clear mockery.",
            "0.2-0.4": "Mild: informal or playful cues that could be read as light teasing.",
            "0.4-0.6": "Moderate: noticeable snark, irony, or dismissiveness.",
            "0.6-0.8": "High: ridicule or dunking tone likely to make the post look mean-spirited.",
            "0.8-1.0": "Very high: explicit mockery, humiliation, or derisive framing.",
        },
    },
    "Overconfident Judgment": {
        "probability_meaning": "How likely readers are to perceive a personal view as stated like an unquestionable fact.",
        "severity_meaning": "How damaging the overcertainty would be if readers challenge the claim.",
        "levels": {
            "0.0-0.2": "Very low: clearly framed as personal, tentative, or evidence-aware.",
            "0.2-0.4": "Mild: somewhat firm opinion but still limited in scope.",
            "0.4-0.6": "Moderate: strong claim with little nuance or support.",
            "0.6-0.8": "High: absolute judgment that invites correction or pushback.",
            "0.8-1.0": "Very high: sweeping certainty presented as fact despite likely contestability.",
        },
    },
    "Context Inappropriateness": {
        "probability_meaning": "How likely the post feels unsuitable for the stated context and audience.",
        "severity_meaning": "How damaging the mismatch would be in that context.",
        "levels": {
            "0.0-0.2": "Very low: tone and content fit the context.",
            "0.2-0.4": "Mild: slightly casual, blunt, or underexplained for the context.",
            "0.4-0.6": "Moderate: context mismatch may distract from the intended point.",
            "0.6-0.8": "High: wording is likely inappropriate for the audience or channel.",
            "0.8-1.0": "Very high: severe mismatch that could trigger public, professional, or relational consequences.",
        },
    },
    "Misinterpretability": {
        "probability_meaning": "How likely readers are to misunderstand the intent, target, or tone.",
        "severity_meaning": "How damaging the misunderstanding would be if it occurs.",
        "levels": {
            "0.0-0.2": "Very low: clear intent, target, and tone.",
            "0.2-0.4": "Mild: minor ambiguity but likely recoverable from context.",
            "0.4-0.6": "Moderate: short, vague, or blunt enough to invite negative readings.",
            "0.6-0.8": "High: likely to be read as harsher, broader, or more hostile than intended.",
            "0.8-1.0": "Very high: ambiguous in a way that strongly invites harmful or hostile interpretation.",
        },
    },
    "Norm Violation": {
        "probability_meaning": "How likely readers are to perceive a social, dignity, fairness, or anti-harassment norm being crossed.",
        "severity_meaning": "How damaging the perceived norm violation would be.",
        "levels": {
            "0.0-0.2": "Very low: no clear dignity, fairness, discrimination, or harassment concern.",
            "0.2-0.4": "Mild: somewhat rude or norm-adjacent but not clearly crossing a boundary.",
            "0.4-0.6": "Moderate: wording may be seen as demeaning or socially unacceptable.",
            "0.6-0.8": "High: clear social norm boundary risk or dignity attack.",
            "0.8-1.0": "Very high: severe demeaning, discriminatory, dehumanizing, or harassment-like language.",
        },
    },
}

DIMENSION_WEIGHTS = {
    "Aggression": 1.35,
    "Group Generalization": 1.2,
    "Sarcasm / Mockery": 1.1,
    "Overconfident Judgment": 0.95,
    "Context Inappropriateness": 1.15,
    "Misinterpretability": 1.0,
    "Norm Violation": 1.4,
}

FALLBACK_CAUTION_THRESHOLD = 45
FALLBACK_HIGH_THRESHOLD = 70
DIMENSION_ATTENTION_PERCENTILE = 60

# Human-designed context multipliers
CONTEXT_RISK_MULTIPLIERS = {
    "public_social": {
        "Aggression": 1.12,
        "Group Generalization": 1.08,
        "Sarcasm / Mockery": 1.15,
        "Overconfident Judgment": 1.05,
        "Context Inappropriateness": 1.25,
        "Misinterpretability": 1.15,
        "Norm Violation": 1.10,
    },
    "workplace": {
        "Aggression": 1.20,
        "Group Generalization": 1.12,
        "Sarcasm / Mockery": 1.10,
        "Overconfident Judgment": 1.12,
        "Context Inappropriateness": 1.35,
        "Misinterpretability": 1.18,
        "Norm Violation": 1.22,
    },
    "private_chat": {
        "Aggression": 0.82,
        "Group Generalization": 0.90,
        "Sarcasm / Mockery": 0.78,
        "Overconfident Judgment": 0.88,
        "Context Inappropriateness": 0.65,
        "Misinterpretability": 0.80,
        "Norm Violation": 0.92,
    },
}

CATEGORY_LABELS = {
    "Aggression": "Aggressive phrasing",
    "Group Generalization": "Sweeping generalization",
    "Sarcasm / Mockery": "Mocking tone",
    "Overconfident Judgment": "Absolute judgment",
    "Context Inappropriateness": "Public-context mismatch",
    "Misinterpretability": "Easy to misread",
    "Norm Violation": "Norm boundary risk",
}


EMOTICON_PATTERN = re.compile(r"(?::|;|=|8|x|X)(?:-|\^|')?(?:\)|\(|D|P|p|/|\\|\|)")
EMOJI_PATTERN = re.compile("[\U0001F300-\U0001FAFF]")
HEDGE_PATTERN = re.compile(r"\b(i think|i feel|maybe|perhaps|kind of|it seems|in my opinion)\b", re.IGNORECASE)
PUBLIC_CONTEXT_PATTERN = re.compile(r"\b(public|sns|social|message|instagram|x|twitter|community|forum|tiktok|comment section)\b", re.IGNORECASE)
VAGUE_TARGET_PATTERN = re.compile(r"\b(they|them|people|everyone|those|that|this)\b", re.IGNORECASE)
SLANG_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_']{1,7}")
REPEATED_TOKEN_PATTERN = re.compile(r"(.)\1{2,}", re.IGNORECASE)
COMMON_FUNCTION_TOKENS = {
    "this", "that", "they", "them", "with", "from", "have", "just", "really", "very",
    "what", "when", "where", "were", "was", "are", "your", "about", "there", "here",
    "place", "people", "because", "would", "could", "should", "maybe", "think", "feel",
}

DIMENSION_PROTOTYPES = {
    "Aggression": [
        "This sounds openly hostile and harsh.",
        "The post attacks the target in a blunt and aggressive way.",
        "This wording feels insulting and confrontational.",
    ],
    "Group Generalization": [
        "The post makes a sweeping claim about a whole group.",
        "This sounds like a broad generalization about many people.",
        "The wording treats a group as if everyone in it is the same.",
    ],
    "Sarcasm / Mockery": [
        "The tone sounds sarcastic, dismissive, or mocking.",
        "This post seems written to ridicule someone.",
        "The wording has a sneering or eye-rolling tone.",
    ],
    "Overconfident Judgment": [
        "The post presents a personal opinion like an unquestionable fact.",
        "This sounds overly certain and leaves little room for nuance.",
        "The wording feels absolute rather than tentative.",
    ],
    "Context Inappropriateness": [
        "This could feel inappropriate in a public social media setting.",
        "The same statement may draw criticism when posted publicly.",
        "This wording may not fit a broad online audience.",
    ],
    "Misinterpretability": [
        "The message is easy to misunderstand and may sound harsher than intended.",
        "This post is ambiguous enough that readers may infer a hostile meaning.",
        "The target or intent is unclear, which raises the chance of misreading.",
    ],
    "Norm Violation": [
        "The tone risks sounding demeaning, dehumanizing, or socially unacceptable.",
        "This wording may cross a social norm boundary.",
        "The post could be read as attacking someone's dignity or status.",
    ],
}

@dataclass
class PipelineOutput:
    context: str
    recommended_context: str
    audience_profile: str
    original_text: str
    normalized_text: str
    tokens: List[str]
    embedding_model: str
    sentence_embedding_preview: List[float]
    contextual_embedding_preview: List[float]
    heuristic_dimensions: Dict[str, float]
    llm_dimension_scores: Dict[str, Dict[str, float]]
    final_dimensions: Dict[str, float]
    validation_report: Dict[str, object]
    heuristic_risk_score: int
    llm_risk_score: Optional[int]
    final_risk_score: int
    risk_level: str
    threshold_type: str
    issue_categories: List[str]
    special_situations: List[str]
    problematic_phrases: List[str]
    explanation_points: List[str]
    llm_backend: str
    llm_prompt: str
    rewrite_suggestion: str


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def decode_web_input_text(text: str) -> str:
    decoded = str(text)
    for _ in range(3):
        next_decoded = html.unescape(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    return decoded


def is_preserved_text_character(char: str) -> bool:
    return bool(re.fullmatch(r"[\w\s!?.,'@#:/()~+\-]", char)) or bool(EMOJI_PATTERN.fullmatch(char))


def preserve_and_clean_special_characters(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = "".join(char if is_preserved_text_character(char) else " " for char in text)
    text = re.sub(r"([!?.,~]){3,}", r"\1\1", text)
    return normalize_whitespace(text)


def reduce_repeated_characters(text: str, keep: int = 2) -> str:
    return re.sub(r"(.)\1{2,}", lambda match: match.group(1) * keep, text)


def preprocess_text(text: str) -> str:
    text = decode_web_input_text(text)
    text = normalize_whitespace(text)
    text = preserve_and_clean_special_characters(text)
    text = reduce_repeated_characters(text)
    return text


def tokenize_text(text: str) -> List[str]:
    token_pattern = r"#[\w_]+|@[\w_]+|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[!?.,]+"
    return re.findall(token_pattern, text.lower())


def contains_public_context(context: str) -> bool:
    return bool(PUBLIC_CONTEXT_PATTERN.search(context or ""))


def extract_candidate_cues(text: str, tokens: List[str]) -> List[Dict[str, str]]:
    text = decode_web_input_text(text)
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_']*|[!?.,]+", text)
    candidates: List[Dict[str, str]] = []
    seen = set()

    def add_cue(cue: str, cue_type: str, source: str) -> None:
        normalized = cue.strip()
        if not normalized:
            return
        key = (normalized.lower(), cue_type)
        if key in seen:
            return
        seen.add(key)
        candidates.append({"cue": normalized, "cue_type": cue_type, "source": source})

    for match in EMOJI_PATTERN.finditer(text):
        add_cue(match.group(0), "emoji", "unicode_emoji")

    for match in EMOTICON_PATTERN.finditer(text):
        add_cue(match.group(0), "emoticon", "ascii_emoticon")

    for token in raw_tokens:
        lowered = token.lower()
        if token.isalpha() and len(token) >= 2 and token.isupper():
            add_cue(token, "all_caps", "capital_emphasis")
            continue
        if REPEATED_TOKEN_PATTERN.search(token) and any(char.isalpha() for char in token):
            add_cue(token, "elongated_token", "repeated_characters")
            continue
        if (
            token.isalpha()
            and 2 <= len(token) <= 5
            and SLANG_TOKEN_PATTERN.fullmatch(token)
            and lowered not in COMMON_FUNCTION_TOKENS
        ):
            vowel_count = sum(char in "aeiou" for char in lowered)
            if vowel_count <= 2 or len(set(lowered)) < len(lowered):
                add_cue(token, "slang_candidate", "short_informal_token")

    if len(tokens) <= 6 and raw_tokens:
        add_cue("short_post", "shape_signal", "post_length")

    return candidates


def clipped_dimension(value: float, low: float = 0.38, high: float = 0.62) -> int:
    if value >= high:
        return 2
    if value >= low:
        return 1
    return 0


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# Model-based semantic similarity
class HybridEmbedder:
    def __init__(self, seed_texts: Optional[List[str]] = None):
        prototype_texts = [item for values in DIMENSION_PROTOTYPES.values() for item in values]
        self.seed_texts = seed_texts or [
            "the visit was not ideal",
            "the situation felt frustrating",
            "i disagree but i may be wrong",
            "the service was not my style",
            "why are people always like this",
        ] + prototype_texts
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf.fit(self.seed_texts)
        self.model_name = "tfidf-fallback"
        self.sentence_model = None

        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                sentence_device = "cuda" if RUNTIME_DEVICE == "cuda" else "cpu"
                self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=sentence_device)
                self.model_name = f"sentence-transformers/all-MiniLM-L6-v2 [{sentence_device}]"
            except Exception:
                self.sentence_model = None

    def encode_sentence(self, text: str) -> np.ndarray:
        if self.sentence_model is not None:
            return np.asarray(self.sentence_model.encode(text), dtype=float)
        return self.tfidf.transform([text]).toarray()[0]

    def encode_contextual(self, text: str, context: str, audience_profile: str) -> np.ndarray:
        contextual_text = f"Context: {context}\nAudience: {audience_profile}\nPost: {text}"
        return self.encode_sentence(contextual_text)


class SemanticRiskScorer:
    def __init__(self, embedder: HybridEmbedder):
        self.embedder = embedder
        self.prototype_embeddings = {
            name: [self.embedder.encode_sentence(example) for example in examples]
            for name, examples in DIMENSION_PROTOTYPES.items()
        }

    def prototype_score(self, sentence_embedding: np.ndarray, contextual_embedding: np.ndarray, dimension: str) -> float:
        similarities = []
        for prototype_embedding in self.prototype_embeddings[dimension]:
            sentence_score = cosine_similarity(sentence_embedding, prototype_embedding)
            contextual_score = cosine_similarity(contextual_embedding, prototype_embedding)
            similarities.append((0.55 * sentence_score) + (0.45 * contextual_score))
        return max(similarities) if similarities else 0.0


def aggregate_cue_score(cue_interpretations: List[Dict[str, object]], key: str, mode: str = "mean") -> float:
    values = [float(item.get(key, 0.0)) for item in cue_interpretations]
    if not values:
        return 0.0
    if mode == "max":
        return float(max(values))
    return float(sum(values) / len(values))


def context_bucket(context: str) -> str:
    lowered = context.lower()
    if re.search(r"\b(private|friend|friends|dm|direct message|chat|kakao|text)\b", lowered):
        return "private_chat"
    if re.search(r"\b(company|work|workplace|office|email|slack|teams|school|class)\b", lowered):
        return "workplace"
    if contains_public_context(context):
        return "public_social"
    return "public_social"


def get_context_risk_multipliers(context: str) -> Dict[str, float]:
    bucket = context_bucket(context)
    return CONTEXT_RISK_MULTIPLIERS.get(bucket, {name: 1.0 for name in DIMENSION_ORDER})


# Human-designed context multipliers
def apply_context_multipliers(scores: Dict[str, float], context: str) -> Dict[str, float]:
    multipliers = get_context_risk_multipliers(context)
    return {
        name: float(np.clip(scores[name] * multipliers.get(name, 1.0), 0.0, 1.0))
        for name in DIMENSION_ORDER
    }


def measure_surface_signals(text: str, tokens: List[str], context: str, cue_analysis: Dict[str, object]) -> Dict[str, float]:
    lowered = text.lower()
    cue_interpretations = cue_analysis.get("cue_interpretations", [])
    summary = cue_analysis.get("summary", {})

    exclamations = min(text.count("!"), 3) / 3
    question_bursts = min(text.count("?"), 3) / 3
    repeated_punctuation = 1.0 if re.search(r"([!?.,~])\1", text) else 0.0
    emoji_density = min(sum(item.get("cue_type") in {"emoji", "emoticon", "slang_candidate", "elongated_token", "all_caps"} for item in cue_interpretations), 3) / 3
    hedge_density = max(min(len(HEDGE_PATTERN.findall(text)), 2) / 2, float(summary.get("hedge_language", 0.0)))
    public_context = 1.0 if contains_public_context(context) else 0.0
    short_post = 1.0 if len(tokens) <= 6 else 0.0
    vague_target = max(1.0 if VAGUE_TARGET_PATTERN.search(lowered) else 0.0, float(summary.get("vague_targeting", 0.0)))
    aggressive_signal = max(aggregate_cue_score(cue_interpretations, "aggression"), aggregate_cue_score(cue_interpretations, "direct_attack", mode="max"))
    group_signal = max(aggregate_cue_score(cue_interpretations, "group_targeting"), float(summary.get("group_generalization", 0.0)))
    sarcasm_signal = max(
        aggregate_cue_score(cue_interpretations, "sarcasm"),
        aggregate_cue_score(cue_interpretations, "mockery"),
        aggregate_cue_score(cue_interpretations, "dismissive", mode="max"),
    )
    norm_signal = aggregate_cue_score(cue_interpretations, "norm_violation")
    direct_attack = aggregate_cue_score(cue_interpretations, "direct_attack", mode="max")
    dismissive_marker = aggregate_cue_score(cue_interpretations, "dismissive", mode="max")
    softening_signal = aggregate_cue_score(cue_interpretations, "softening")
    sentiment_targeting = max(direct_attack, aggressive_signal * vague_target)
    needs_web_lookup = 1.0 if any(bool(item.get("needs_web_lookup")) for item in cue_interpretations) else 0.0

    return {
        "exclamations": exclamations,
        "question_bursts": question_bursts,
        "repeated_punctuation": repeated_punctuation,
        "emoji_density": emoji_density,
        "hedge_density": hedge_density,
        "public_context": public_context,
        "short_post": short_post,
        "vague_target": vague_target,
        "aggressive_signal": aggressive_signal,
        "group_signal": group_signal,
        "sarcasm_signal": sarcasm_signal,
        "norm_signal": norm_signal,
        "direct_attack": direct_attack,
        "dismissive_marker": dismissive_marker,
        "softening_signal": softening_signal,
        "sentiment_targeting": sentiment_targeting,
        "needs_web_lookup": needs_web_lookup,
    }


# Final score aggregation starts with first-pass heuristic scores
def first_pass_risk_classification(
    text: str,
    tokens: List[str],
    context: str,
    sentence_embedding: np.ndarray,
    contextual_embedding: np.ndarray,
    scorer: SemanticRiskScorer,
    cue_analysis: Dict[str, object],
) -> Dict[str, float]:
    semantic_scores = {
        name: scorer.prototype_score(sentence_embedding, contextual_embedding, name)
        for name in DIMENSION_ORDER
    }
    surface = measure_surface_signals(text, tokens, context, cue_analysis)

    scores = {
        "Aggression": semantic_scores["Aggression"] + 0.26 * surface["aggressive_signal"] + 0.24 * surface["direct_attack"] + 0.16 * surface["sentiment_targeting"] + 0.08 * surface["exclamations"] + 0.08 * surface["repeated_punctuation"],
        "Group Generalization": semantic_scores["Group Generalization"] + 0.18 * surface["group_signal"] + 0.10 * surface["vague_target"],
        "Sarcasm / Mockery": semantic_scores["Sarcasm / Mockery"] + 0.20 * surface["sarcasm_signal"] + 0.08 * surface["emoji_density"] + 0.05 * surface["question_bursts"],
        "Overconfident Judgment": semantic_scores["Overconfident Judgment"] - 0.12 * surface["hedge_density"] - 0.08 * surface["softening_signal"] + 0.06 * (1.0 - surface["hedge_density"]),
        "Context Inappropriateness": semantic_scores["Context Inappropriateness"] + 0.15 * surface["public_context"] + 0.08 * surface["dismissive_marker"] + 0.05 * surface["short_post"] + 0.04 * surface["needs_web_lookup"],
        "Misinterpretability": semantic_scores["Misinterpretability"] + 0.10 * surface["short_post"] + 0.10 * surface["vague_target"] + 0.06 * surface["dismissive_marker"] + 0.05 * surface["needs_web_lookup"],
        "Norm Violation": semantic_scores["Norm Violation"] + 0.18 * surface["norm_signal"] + 0.05 * surface["public_context"],
    }

    if surface["direct_attack"] and surface["dismissive_marker"]:
        scores["Aggression"] = max(scores["Aggression"], 0.58 - 0.22 * surface["hedge_density"])
        scores["Sarcasm / Mockery"] = max(scores["Sarcasm / Mockery"], 0.52)
        scores["Context Inappropriateness"] = max(scores["Context Inappropriateness"], 0.46 - 0.08 * surface["hedge_density"])
    elif surface["direct_attack"]:
        scores["Aggression"] = max(scores["Aggression"], 0.48 - 0.18 * surface["hedge_density"])
    elif surface["aggressive_signal"] and surface["dismissive_marker"]:
        scores["Aggression"] = max(scores["Aggression"], 0.40 - 0.12 * surface["hedge_density"])
        scores["Sarcasm / Mockery"] = max(scores["Sarcasm / Mockery"], 0.42)

    scores = {name: float(np.clip(value, 0.0, 1.0)) for name, value in scores.items()}
    if (surface["hedge_density"] + surface["softening_signal"]) >= 0.6 and surface["direct_attack"]:
        scores["Aggression"] = max(0.0, scores["Aggression"] - 0.18)
        scores["Context Inappropriateness"] = max(0.0, scores["Context Inappropriateness"] - 0.12)

    return apply_context_multipliers(scores, context)


def calculate_weighted_risk_score(risk_dimensions: Dict[str, float]) -> int:
    weighted_sum = sum(risk_dimensions[name] * DIMENSION_WEIGHTS[name] for name in DIMENSION_ORDER)
    max_sum = sum(DIMENSION_WEIGHTS[name] for name in DIMENSION_ORDER)
    score = round((weighted_sum / max_sum) * 100)
    return int(np.clip(score, 0, 100))


def dimension_attention_threshold(risk_dimensions: Dict[str, float]) -> float:
    values = [float(risk_dimensions[name]) for name in DIMENSION_ORDER]
    if not values:
        return 1.0
    return float(np.percentile(values, DIMENSION_ATTENTION_PERCENTILE))


def derive_issue_categories(risk_dimensions: Dict[str, float]) -> List[str]:
    threshold = dimension_attention_threshold(risk_dimensions)
    categories = [
        CATEGORY_LABELS[name]
        for name in DIMENSION_ORDER
        if risk_dimensions[name] >= threshold and risk_dimensions[name] > 0.0
    ]
    return categories or ["Low explicit risk"]


def top_dimensions(risk_dimensions: Dict[str, float], limit: int = 3) -> List[str]:
    threshold = dimension_attention_threshold(risk_dimensions)
    ranked = sorted(risk_dimensions.items(), key=lambda item: item[1], reverse=True)
    return [name for name, score in ranked if score >= threshold and score > 0.0][:limit]


def calibrate_thresholds(validation_scores: List[float]) -> Dict[str, float]:
    if not validation_scores:
        return {"caution": FALLBACK_CAUTION_THRESHOLD, "high": FALLBACK_HIGH_THRESHOLD, "source": "fallback_defaults"}
    clipped_scores = [float(np.clip(score, 0, 100)) for score in validation_scores]
    return {
        "caution": float(np.percentile(clipped_scores, 60)),
        "high": float(np.percentile(clipped_scores, 85)),
        "source": "calibrated_percentiles",
    }


def risk_level_from_score(score: int, thresholds: Optional[Dict[str, float]] = None) -> str:
    active_thresholds = thresholds or {
        "caution": FALLBACK_CAUTION_THRESHOLD,
        "high": FALLBACK_HIGH_THRESHOLD,
        "source": "fallback_defaults",
    }
    source_suffix = "calibrated thresholds" if thresholds else "fallback default thresholds"
    if score >= active_thresholds["high"]:
        return f"High ({source_suffix})"
    if score >= active_thresholds["caution"]:
        return f"Caution ({source_suffix})"
    return f"Low ({source_suffix})"


def threshold_type(thresholds: Optional[Dict[str, float]] = None) -> str:
    return "calibrated_percentiles" if thresholds else "fallback_defaults"


def extract_first_json_block(text: str) -> Optional[dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def clip01(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(np.clip(value, 0.0, 1.0))
    return 0.0


def empty_cue_analysis(candidate_cues: Optional[List[Dict[str, str]]] = None) -> Dict[str, object]:
    return {
        "candidate_cues": candidate_cues or [],
        "cue_interpretations": [],
        "summary": {
            "vague_targeting": 0.0,
            "group_generalization": 0.0,
            "hedge_language": 0.0,
        },
        "backend": "disabled",
    }


def fallback_cue_analysis(candidate_cues: Optional[List[Dict[str, str]]] = None, backend: str = "fallback-structure") -> Dict[str, object]:
    analysis = empty_cue_analysis(candidate_cues)
    analysis["backend"] = backend
    structural_cues = [item for item in candidate_cues or []]
    analysis["cue_interpretations"] = [
        {
            "cue": str(item.get("cue", "")),
            "cue_type": str(item.get("cue_type", "unknown")),
            "sarcasm": 0.0,
            "mockery": 0.0,
            "softening": 0.0,
            "aggression": 0.0,
            "dismissive": 0.0,
            "direct_attack": 0.0,
            "group_targeting": 0.0,
            "norm_violation": 0.0,
            "confidence": 0.20,
            "needs_web_lookup": False,
            "lookup_query": "",
            "reason": "structure-only fallback; lexical meaning requires LLM interpretation",
        }
        for item in structural_cues
    ]
    return analysis


def build_cue_interpretation_prompt(
    context: str,
    original_text: str,
    normalized_text: str,
    tokens: List[str],
    candidate_cues: List[Dict[str, str]],
) -> str:
    cue_lines = "\n".join(
        f"- cue={item['cue']} | type={item['cue_type']} | source={item['source']}"
        for item in candidate_cues
    )
    return f"""
You are interpreting fast-changing social-media cues inside a short post.

Context: {context}
Original text: {original_text}
Normalized text: {normalized_text}
Tokens: {tokens}
Candidate cues:
{cue_lines}

Return ONLY valid JSON with this schema:
{{
  "cue_interpretations": [
    {{
      "cue": "string",
      "cue_type": "string",
      "relevant": true,
      "sarcasm": 0.0,
      "mockery": 0.0,
      "softening": 0.0,
      "aggression": 0.0,
      "dismissive": 0.0,
      "direct_attack": 0.0,
      "group_targeting": 0.0,
      "norm_violation": 0.0,
      "confidence": 0.0,
      "needs_web_lookup": false,
      "lookup_query": "string",
      "reason": "string"
    }}
  ],
  "summary": {{
    "vague_targeting": 0.0,
    "group_generalization": 0.0,
    "hedge_language": 0.0
  }}
}}

Rules:
- Use 0.0 to 1.0 for every score.
- Mark relevant=false if a candidate cue is probably not meaningful in this context.
- If the cue looks recent, ambiguous, or culturally unstable, set needs_web_lookup=true.
- Judge the cue inside this exact sentence, not in isolation.
""".strip()


def sanitize_cue_analysis(payload: Optional[dict], candidate_cues: List[Dict[str, str]], backend: str) -> Dict[str, object]:
    sanitized = empty_cue_analysis(candidate_cues)
    sanitized["backend"] = backend
    if not payload:
        return sanitized

    cleaned_interpretations = []
    for item in payload.get("cue_interpretations", []):
        if not isinstance(item, dict):
            continue
        cue = str(item.get("cue", "")).strip()
        if not cue:
            continue
        if item.get("relevant", True) is False:
            continue
        cleaned_interpretations.append(
            {
                "cue": cue,
                "cue_type": str(item.get("cue_type", "unknown")),
                "sarcasm": clip01(item.get("sarcasm", 0.0)),
                "mockery": clip01(item.get("mockery", 0.0)),
                "softening": clip01(item.get("softening", 0.0)),
                "aggression": clip01(item.get("aggression", 0.0)),
                "dismissive": clip01(item.get("dismissive", 0.0)),
                "direct_attack": clip01(item.get("direct_attack", 0.0)),
                "group_targeting": clip01(item.get("group_targeting", 0.0)),
                "norm_violation": clip01(item.get("norm_violation", 0.0)),
                "confidence": clip01(item.get("confidence", 0.0)),
                "needs_web_lookup": bool(item.get("needs_web_lookup", False)),
                "lookup_query": str(item.get("lookup_query", "")).strip(),
                "reason": str(item.get("reason", "")).strip(),
            }
        )

    summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {}
    sanitized["cue_interpretations"] = cleaned_interpretations
    sanitized["summary"] = {
        "vague_targeting": clip01(summary.get("vague_targeting", 0.0)),
        "group_generalization": clip01(summary.get("group_generalization", 0.0)),
        "hedge_language": clip01(summary.get("hedge_language", 0.0)),
    }
    return sanitized


# LLM-based contextual judgment
def format_dimension_rubrics() -> str:
    lines = []
    for name in DIMENSION_ORDER:
        rubric = DIMENSION_RUBRICS[name]
        lines.append(f"{name}:")
        lines.append(f"  probability: {rubric['probability_meaning']}")
        lines.append(f"  severity: {rubric['severity_meaning']}")
        for band, description in rubric["levels"].items():
            lines.append(f"  {band}: {description}")
    return "\n".join(lines)


def build_qwen_prompt(
    context: str,
    audience_profile: str,
    original_text: str,
    normalized_text: str,
    tokens: List[str],
    first_pass_dimensions: Dict[str, float],
    cue_analysis: Dict[str, object],
) -> str:
    first_pass_lines = "\n".join(f"- {name}: {score:.2f}" for name, score in first_pass_dimensions.items())
    candidate_summary = ", ".join(f"{item['cue']}<{item['cue_type']}>" for item in cue_analysis.get("candidate_cues", [])) or "None"
    interpreted_cues = cue_analysis.get("cue_interpretations", [])
    interpreted_lines = "\n".join(
        f"- {item['cue']} ({item['cue_type']}): sarcasm={item['sarcasm']:.2f}, mockery={item['mockery']:.2f}, aggression={item['aggression']:.2f}, direct_attack={item['direct_attack']:.2f}, lookup={item['needs_web_lookup']}"
        for item in interpreted_cues[:8]
    ) or "- None"
    context_multipliers = get_context_risk_multipliers(context)
    context_lines = "\n".join(f"- {name}: x{context_multipliers[name]:.2f}" for name in DIMENSION_ORDER)
    rubric_text = format_dimension_rubrics()
    return f"""
You are a rubric-based social-backlash judge for short posts.

Goal: estimate the probability that a post receives criticism, backlash, quote-tweeting, HR concern, or negative replies in the given context.
Do not score from instinct alone. Apply the human-designed rubrics below.

Context: {context}
Audience profile: {audience_profile}
Original text: {original_text}
Normalized text: {normalized_text}
Tokens: {tokens}
Candidate cues: {candidate_summary}
Cue interpretation summary:
{interpreted_lines}

First-pass model-based heuristic scores, already 0.0 to 1.0:
{first_pass_lines}

Human-designed context multipliers used by the heuristic layer:
{context_lines}

Human-designed scoring rubrics:
{rubric_text}

Return ONLY valid JSON with this schema:
{{
  "backlash_probability": 0.0,
  "dimension_scores": {{
    "Aggression": {{"probability": 0.0, "severity": 0.0, "confidence": 0.0, "rubric_reason": "string"}},
    "Group Generalization": {{"probability": 0.0, "severity": 0.0, "confidence": 0.0, "rubric_reason": "string"}},
    "Sarcasm / Mockery": {{"probability": 0.0, "severity": 0.0, "confidence": 0.0, "rubric_reason": "string"}},
    "Overconfident Judgment": {{"probability": 0.0, "severity": 0.0, "confidence": 0.0, "rubric_reason": "string"}},
    "Context Inappropriateness": {{"probability": 0.0, "severity": 0.0, "confidence": 0.0, "rubric_reason": "string"}},
    "Misinterpretability": {{"probability": 0.0, "severity": 0.0, "confidence": 0.0, "rubric_reason": "string"}},
    "Norm Violation": {{"probability": 0.0, "severity": 0.0, "confidence": 0.0, "rubric_reason": "string"}}
  }},
  "special_situations": ["string"],
  "problematic_phrases": ["string"],
  "explanations": ["string", "string", "string"],
  "rewrite": "string"
}}

Rules:
- probability, severity, confidence, and backlash_probability must be between 0.0 and 1.0.
- probability means how likely the audience will perceive that dimension.
- severity means how damaging that perception would be if noticed.
- confidence means how reliable your judgment is from this wording and context.
- rubric_reason must briefly identify the applied rubric band and why.
- Keep explanations concrete and audience-facing.
- The rewrite must preserve intent while lowering backlash risk.
- Return only valid JSON.
""".strip()


class QwenContextAnalyzer:
    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self._transformers_tokenizer = None
        self._transformers_model = None
        self._transformers_reason = ""

    def _can_use_ollama(self) -> bool:
        tags_url = QWEN_OLLAMA_URL.replace("/api/generate", "/api/tags")
        try:
            with urllib.request.urlopen(tags_url, timeout=5) as response:
                if response.status != 200:
                    return False
                data = json.loads(response.read().decode("utf-8"))
                models = data.get("models", []) if isinstance(data, dict) else []
                names = {str(item.get("name", "")) for item in models if isinstance(item, dict)}
                return not names or QWEN_OLLAMA_MODEL in names
        except Exception:
            return False

    def _generate_with_ollama(self, prompt: str) -> Optional[dict]:
        payload = {
            "model": QWEN_OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }
        req = urllib.request.Request(QWEN_OLLAMA_URL, method="POST")
        try:
            with urllib.request.urlopen(req, data=json.dumps(payload).encode("utf-8"), timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                return extract_first_json_block(data.get("response", ""))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None

    def _load_transformers(self) -> bool:
        if not TRANSFORMERS_AVAILABLE:
            self._transformers_reason = "transformers-not-installed"
            return False
        if self._transformers_model is not None and self._transformers_tokenizer is not None:
            return True
        if not torch_cuda_available():
            self._transformers_reason = "cuda-not-available"
            return False
        if GPU_MEMORY_GB < MIN_QWEN_TRANSFORMERS_VRAM_GB:
            self._transformers_reason = f"insufficient-vram-{GPU_MEMORY_GB}GB"
            return False
        try:
            self._transformers_tokenizer = AutoTokenizer.from_pretrained(QWEN_TRANSFORMERS_MODEL)
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
            self._transformers_model = AutoModelForCausalLM.from_pretrained(QWEN_TRANSFORMERS_MODEL, **model_kwargs)
            return True
        except Exception:
            self._transformers_model = None
            self._transformers_tokenizer = None
            self._transformers_reason = "transformers-load-failed"
            return False

    def _generate_with_transformers(self, prompt: str) -> Optional[dict]:
        if not self._load_transformers():
            return None
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self._transformers_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._transformers_tokenizer(text, return_tensors="pt")
            if hasattr(self._transformers_model, "device"):
                inputs = {key: value.to(self._transformers_model.device) for key, value in inputs.items()}
            output = self._transformers_model.generate(**inputs, max_new_tokens=400, do_sample=False)
            generated = output[0][inputs["input_ids"].shape[-1]:]
            decoded = self._transformers_tokenizer.decode(generated, skip_special_tokens=True)
            return extract_first_json_block(decoded)
        except Exception:
            return None

    def analyze(self, prompt: str) -> Dict[str, object]:
        if self.backend == "disabled":
            return {"backend": "disabled", "result": None}
        if self.backend in {"auto", "ollama"} and self._can_use_ollama():
            result = self._generate_with_ollama(prompt)
            if result is not None:
                return {"backend": "ollama", "result": result}
        if self.backend in {"auto", "transformers"}:
            result = self._generate_with_transformers(prompt)
            if result is not None:
                return {"backend": "transformers", "result": result}
            if self.backend == "transformers" and self._transformers_reason:
                return {"backend": f"transformers-skipped:{self._transformers_reason}", "result": None}
        return {"backend": "unavailable", "result": None}


def interpret_cues_with_llm(
    context: str,
    original_text: str,
    normalized_text: str,
    tokens: List[str],
    candidate_cues: List[Dict[str, str]],
    analyzer: QwenContextAnalyzer,
) -> Dict[str, object]:
    if not candidate_cues:
        return empty_cue_analysis(candidate_cues)

    prompt = build_cue_interpretation_prompt(
        context=context,
        original_text=original_text,
        normalized_text=normalized_text,
        tokens=tokens,
        candidate_cues=candidate_cues,
    )
    response = analyzer.analyze(prompt)
    sanitized = sanitize_cue_analysis(response.get("result"), candidate_cues, str(response.get("backend", "unavailable")))
    if not sanitized.get("cue_interpretations"):
        return fallback_cue_analysis(candidate_cues, backend=str(response.get("backend", "fallback-structure")))
    return sanitized


# Validation layer
def sanitize_llm_dimension_scores(payload: Optional[dict]) -> Dict[str, Dict[str, object]]:
    scores = {
        name: {"probability": 0.0, "severity": 0.0, "confidence": 0.0, "rubric_reason": "missing", "available": 0.0}
        for name in DIMENSION_ORDER
    }
    if not payload:
        return scores

    raw_scores = payload.get("dimension_scores", {})
    if isinstance(raw_scores, dict):
        for name in DIMENSION_ORDER:
            raw_item = raw_scores.get(name, {})
            if isinstance(raw_item, dict):
                scores[name] = {
                    "probability": clip01(raw_item.get("probability", 0.0)),
                    "severity": clip01(raw_item.get("severity", 0.0)),
                    "confidence": clip01(raw_item.get("confidence", 0.0)),
                    "rubric_reason": str(raw_item.get("rubric_reason", "")).strip(),
                    "available": 1.0,
                }

    legacy_adjustments = payload.get("adjustments", {})
    if isinstance(legacy_adjustments, dict) and not any(item["available"] for item in scores.values()):
        for name in DIMENSION_ORDER:
            value = legacy_adjustments.get(name, 0)
            if isinstance(value, (int, float)):
                probability = np.clip(0.5 + (0.22 * float(value)), 0.0, 1.0)
                scores[name] = {
                    "probability": float(probability),
                    "severity": float(probability),
                    "confidence": 0.35,
                    "rubric_reason": "legacy adjustment converted to probability; rubric reason unavailable",
                    "available": 1.0,
                }
    return scores


def validate_llm_scores(
    heuristic_dimensions: Dict[str, float],
    llm_dimension_scores: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    warnings = []
    dimension_flags = {name: [] for name in DIMENSION_ORDER}

    for name in DIMENSION_ORDER:
        score = llm_dimension_scores.get(name)
        if not score or not score.get("available", 0.0):
            dimension_flags[name].append("missing_dimension")
            warnings.append(f"{name}: missing LLM dimension score")
            continue

        probability = score.get("probability")
        severity = score.get("severity")
        confidence = score.get("confidence")
        for key, value in {"probability": probability, "severity": severity, "confidence": confidence}.items():
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                dimension_flags[name].append("out_of_range")
                warnings.append(f"{name}: {key} is outside [0, 1]")

        probability = float(np.clip(probability if isinstance(probability, (int, float)) else 0.0, 0.0, 1.0))
        severity = float(np.clip(severity if isinstance(severity, (int, float)) else 0.0, 0.0, 1.0))
        confidence = float(np.clip(confidence if isinstance(confidence, (int, float)) else 0.0, 0.0, 1.0))
        heuristic_score = float(heuristic_dimensions.get(name, 0.0))
        llm_risk = (0.70 * probability) + (0.30 * severity)

        if abs(llm_risk - heuristic_score) > 0.35:
            dimension_flags[name].append("large_disagreement")
            warnings.append(f"{name}: LLM risk differs substantially from heuristic score")
        if confidence < 0.40:
            dimension_flags[name].append("low_confidence")
            warnings.append(f"{name}: LLM confidence is low")
        if severity >= 0.80 and probability <= 0.30:
            dimension_flags[name].append("high_severity_low_probability")
            warnings.append(f"{name}: severe if noticed, but probability is low")

    return {
        "is_valid": not any(flags for flags in dimension_flags.values()),
        "warnings": list(dict.fromkeys(warnings)),
        "dimension_flags": {name: flags for name, flags in dimension_flags.items() if flags},
    }


# Final score aggregation
def merge_dimension_scores(
    heuristic_dimensions: Dict[str, float],
    llm_dimension_scores: Dict[str, Dict[str, object]],
    validation_report: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    merged = {}
    flags_by_dimension = (validation_report or {}).get("dimension_flags", {})
    for name in DIMENSION_ORDER:
        heuristic_score = float(heuristic_dimensions.get(name, 0.0))
        llm_score = llm_dimension_scores.get(name, {})
        if not llm_score.get("available", 0.0):
            merged[name] = heuristic_score
            continue

        probability = float(llm_score.get("probability", 0.0))
        severity = float(llm_score.get("severity", probability))
        confidence = float(llm_score.get("confidence", 0.0))
        llm_risk = (0.70 * probability) + (0.30 * severity)
        base_weight = 0.25 + (0.55 * confidence)
        flags = flags_by_dimension.get(name, [])
        # Fix: validation flags now materially reduce LLM influence.
        if "low_confidence" in flags:
            base_weight *= 0.5
        if "large_disagreement" in flags:
            base_weight *= 0.5
        merged[name] = float(np.clip((1.0 - base_weight) * heuristic_score + base_weight * llm_risk, 0.0, 1.0))
    return merged


def soften_expression(text: str) -> str:
    softened = preserve_and_clean_special_characters(text).strip(" .!?")
    if not softened:
        return "I would phrase this more neutrally."
    softened = softened[0].lower() + softened[1:] if len(softened) > 1 else softened.lower()
    if not re.search(r"\bi\b", softened):
        softened = f"I think {softened}"
    softened = softened.rstrip(".") + "."
    return softened[0].upper() + softened[1:]


def remove_cue_from_text(text: str, cue: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_']+", cue):
        return re.sub(rf"\b{re.escape(cue)}\b", "", text, flags=re.IGNORECASE)
    return text.replace(cue, "")


def cue_risk_score(item: Dict[str, object]) -> float:
    return max(
        float(item.get("mockery", 0.0)),
        float(item.get("dismissive", 0.0)),
        float(item.get("sarcasm", 0.0)),
        float(item.get("aggression", 0.0)),
        float(item.get("direct_attack", 0.0)),
        float(item.get("norm_violation", 0.0)),
    )


def meaningful_word_count(text: str) -> int:
    return len(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def has_negative_opinion_signal(text: str, cues: List[str]) -> bool:
    combined = " ".join([text, *cues]).lower()
    return bool(re.search(r"\b(trash|bad|awful|terrible|worst|sucks?|hate|mid|cringe|lame)\b", combined))


def build_safe_fallback_rewrite(original_text: str, revised_text: str, removed_cues: List[str]) -> str:
    original_text = decode_web_input_text(original_text)
    original_clean = preserve_and_clean_special_characters(original_text).strip(" .!?")
    revised_clean = preserve_and_clean_special_characters(revised_text).strip(" .!?")
    source = revised_clean or original_clean
    if has_negative_opinion_signal(original_clean, removed_cues):
        if re.search(r"\b(this|that)\s+place\b", original_clean, re.IGNORECASE):
            return "I had a disappointing experience with this place."
        if re.search(r"\b(this|that)\s+(was|is)\b", original_clean, re.IGNORECASE):
            return "I did not enjoy this, and I would explain why more clearly."
        return "I had a negative reaction to this, but I would explain it more specifically."
    if source:
        return soften_expression(source)
    return "I would phrase this more neutrally."


def is_useful_recommendation(original_text: str, revised_text: str, removed_cues: List[str]) -> bool:
    revised_clean = preserve_and_clean_special_characters(revised_text).strip(" .!?")
    if meaningful_word_count(revised_clean) < 4 and removed_cues:
        return False
    if has_negative_opinion_signal(original_text, removed_cues) and not has_negative_opinion_signal(revised_clean, []):
        return False
    return bool(revised_clean)


def clean_rewrite_candidate(text: str) -> str:
    cleaned = preserve_and_clean_special_characters(decode_web_input_text(text)).strip(" .!?")
    return cleaned.rstrip(".") + "." if cleaned else ""


def is_acceptable_rewrite_candidate(text: str, cue_analysis: Dict[str, object]) -> bool:
    if not text or detect_problematic_phrases(text, cue_analysis):
        return False
    return meaningful_word_count(text) >= 4


def build_recommended_context(text: str, final_dimensions: Dict[str, float], context: str, cue_analysis: Dict[str, object]) -> str:
    text = decode_web_input_text(text)
    revised = preserve_and_clean_special_characters(text)
    removed_cues: List[str] = []
    attention_threshold = dimension_attention_threshold(final_dimensions)
    public_or_misread_risk = max(
        final_dimensions["Context Inappropriateness"],
        final_dimensions["Misinterpretability"],
    ) >= attention_threshold
    stylistic_cleanup_types = {"slang_candidate", "elongated_token", "all_caps", "emoji", "emoticon"}

    for item in cue_analysis.get("cue_interpretations", []):
        cue = str(item.get("cue", "")).strip()
        cue_type = str(item.get("cue_type", ""))
        semantic_risk = cue_risk_score(item) >= 0.6
        structure_risk = public_or_misread_risk and cue_type in stylistic_cleanup_types
        if cue and cue_type != "shape_signal" and (semantic_risk or structure_risk):
            removed_cues.append(cue)
            revised = remove_cue_from_text(revised, cue)

    revised = re.sub(r"\s+", " ", revised).strip(" .!?")
    if revised and is_useful_recommendation(text, revised, removed_cues):
        revised = re.sub(r"\b(is|are|was|were|be|being|been)\s*$", "", revised, flags=re.IGNORECASE).strip()
        if is_useful_recommendation(text, revised, removed_cues):
            return revised.rstrip(".") + "."
    return build_safe_fallback_rewrite(text, revised, removed_cues)


def detect_problematic_phrases(text: str, cue_analysis: Dict[str, object]) -> List[str]:
    phrases = []
    # Fix: catch structure-level toxicity even when no explicit slur is present.
    structural_patterns = [
        r"\b(?:hate|hates|hated|kill|killing|destroy|trash|stupid|idiot|useless|worthless)\b",
        r"\b(?:they|them|people|everyone|those)\b.{0,24}\b(?:are|is|were|was)\b.{0,24}\b(?:trash|stupid|idiots|awful|useless|worthless)\b",
        r"\b(?:you are|you're|this is|that is|they are|they're)\b.{0,32}\b(?:trash|stupid|awful|useless|worthless|idiotic)\b",
    ]
    for pattern in structural_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrases.append(match.group(0).strip())
    for item in cue_analysis.get("cue_interpretations", []):
        risk = max(
            float(item.get("aggression", 0.0)),
            float(item.get("mockery", 0.0)),
            float(item.get("dismissive", 0.0)),
            float(item.get("direct_attack", 0.0)),
            float(item.get("norm_violation", 0.0)),
        )
        cue = str(item.get("cue", "")).strip()
        if risk >= 0.55 and cue and re.search(rf"\b{re.escape(cue)}\b", text, flags=re.IGNORECASE):
            phrases.append(cue)
    return list(dict.fromkeys([phrase for phrase in phrases if phrase]))


def detect_special_situations(final_dimensions: Dict[str, float], text: str, context: str, cue_analysis: Dict[str, object]) -> List[str]:
    situations = []
    if final_dimensions["Aggression"] >= dimension_attention_threshold(final_dimensions) and contains_public_context(context):
        situations.append("public negative review tone")
    if final_dimensions["Sarcasm / Mockery"] >= dimension_attention_threshold(final_dimensions) and any(max(item.get("mockery", 0.0), item.get("dismissive", 0.0)) >= 0.5 for item in cue_analysis.get("cue_interpretations", [])):
        situations.append("dismissive humor / dunking risk")
    if final_dimensions["Misinterpretability"] >= dimension_attention_threshold(final_dimensions) and len(text.split()) <= 8:
        situations.append("short post prone to hostile reading")
    if any(bool(item.get("needs_web_lookup")) for item in cue_analysis.get("cue_interpretations", [])):
        situations.append("dynamic slang / emoji meaning may need fresh verification")
    return situations


def fallback_explanations(final_dimensions: Dict[str, float], context: str) -> List[str]:
    explanations = []
    if final_dimensions["Aggression"] >= dimension_attention_threshold(final_dimensions):
        explanations.append("The wording lands as harsher than a normal public opinion post, so replies may focus on tone rather than the point.")
    if final_dimensions["Sarcasm / Mockery"] >= dimension_attention_threshold(final_dimensions):
        explanations.append("The sarcastic or dunking tone can make neutral readers read the post as mean-spirited.")
    if final_dimensions["Group Generalization"] >= dimension_attention_threshold(final_dimensions):
        explanations.append("Readers may push back if the post sounds like it is judging a whole group instead of one case.")
    if final_dimensions["Context Inappropriateness"] >= dimension_attention_threshold(final_dimensions):
        explanations.append(f"In a broad public space like {context}, sharp phrasing often attracts criticism faster than in a private chat.")
    if final_dimensions["Misinterpretability"] >= dimension_attention_threshold(final_dimensions):
        explanations.append("Because the statement is short and blunt, people can interpret it more negatively than you intended.")
    return explanations or ["The post is not overtly extreme, but public readers may still react to the tone and framing."]


FINAL_CONTEXT_RISK_BIASES = {
    "public_social": 3,
    "workplace": 8,
    "private_chat": -8,
}


def merge_risk_scores(heuristic_score: int, llm_score: Optional[int], validation_report: Optional[Dict[str, object]] = None) -> int:
    if llm_score is None:
        return heuristic_score
    llm_weight = 0.35
    flags = (validation_report or {}).get("dimension_flags", {})
    flattened_flags = [flag for dimension_flags in flags.values() for flag in dimension_flags]
    # Fix: score-level LLM weight follows validation reliability.
    if "low_confidence" in flattened_flags:
        llm_weight *= 0.5
    if "large_disagreement" in flattened_flags:
        llm_weight *= 0.5
    if (validation_report or {}).get("llm_fallback") or (validation_report or {}).get("llm_zero_scores"):
        llm_weight = 0.0
    return int(round(((1.0 - llm_weight) * heuristic_score) + (llm_weight * llm_score)))


def apply_final_context_adjustment(base_score: int, context: str) -> int:
    bucket = context_bucket(context)
    context_bias = FINAL_CONTEXT_RISK_BIASES.get(bucket, 0)
    context_adjustment = float(np.clip(base_score + context_bias, 0, 100))
    # Fix: context adjusts the final score but cannot dominate it.
    final_score = round((0.8 * base_score) + (0.2 * context_adjustment))
    # Fix: avoid double-penalizing private context into unrealistic near-zero scores.
    final_score = max(final_score, int(np.floor(base_score * 0.7)))
    return int(np.clip(final_score, 0, 100))


def run_backlash_risk_pipeline(text: str, context: str = SOCIAL_CONTEXT, audience_profile: str = AUDIENCE_PROFILE) -> PipelineOutput:
    web_input_text = decode_web_input_text(text)
    normalized_text = preprocess_text(web_input_text)
    tokens = tokenize_text(normalized_text)
    candidate_cues = extract_candidate_cues(web_input_text, tokens)
    # Fix: force an active LLM backend unless the configured backend is already explicit.
    active_backend = QWEN_BACKEND if QWEN_BACKEND != "disabled" else "ollama"
    analyzer = QwenContextAnalyzer(backend=active_backend)
    cue_analysis = interpret_cues_with_llm(
        context=context,
        original_text=web_input_text,
        normalized_text=normalized_text,
        tokens=tokens,
        candidate_cues=candidate_cues,
        analyzer=analyzer,
    )

    embedder = HybridEmbedder()
    sentence_embedding = embedder.encode_sentence(normalized_text)
    contextual_embedding = embedder.encode_contextual(normalized_text, context, audience_profile)
    scorer = SemanticRiskScorer(embedder)

    heuristic_dimensions = first_pass_risk_classification(
        normalized_text,
        tokens,
        context,
        sentence_embedding,
        contextual_embedding,
        scorer,
        cue_analysis,
    )
    heuristic_risk_score = calculate_weighted_risk_score(heuristic_dimensions)

    llm_prompt = build_qwen_prompt(
        context=context,
        audience_profile=audience_profile,
        original_text=web_input_text,
        normalized_text=normalized_text,
        tokens=tokens,
        first_pass_dimensions=heuristic_dimensions,
        cue_analysis=cue_analysis,
    )

    llm_response = analyzer.analyze(llm_prompt)
    llm_payload = llm_response.get("result") if llm_response else None
    llm_backend = str(llm_response.get("backend", "unavailable")) if llm_response else "unavailable"
    llm_dimension_scores = sanitize_llm_dimension_scores(llm_payload)
    validation_report = validate_llm_scores(heuristic_dimensions, llm_dimension_scores)
    if not llm_payload:
        validation_report["warnings"].append("LLM unavailable; using heuristic fallback")
        validation_report["llm_fallback"] = True
    elif not any(score.get("available", 0.0) and (score.get("probability", 0.0) or score.get("severity", 0.0)) for score in llm_dimension_scores.values()):
        validation_report["warnings"].append("LLM returned zero-valued scores; using heuristic-dominant merge")
        validation_report["llm_zero_scores"] = True
    final_dimensions = merge_dimension_scores(heuristic_dimensions, llm_dimension_scores, validation_report)

    llm_risk_score = None
    if llm_payload and isinstance(llm_payload.get("backlash_probability"), (int, float)):
        raw_probability = float(llm_payload["backlash_probability"])
        llm_risk_score = int(np.clip(round(raw_probability * 100 if raw_probability <= 1 else raw_probability), 0, 100))

    base_risk_score = merge_risk_scores(calculate_weighted_risk_score(final_dimensions), llm_risk_score, validation_report)
    final_risk_score = apply_final_context_adjustment(base_risk_score, context)
    risk_level = risk_level_from_score(final_risk_score)
    active_threshold_type = threshold_type()
    issue_categories = derive_issue_categories(final_dimensions)
    special_situations = llm_payload.get("special_situations", []) if llm_payload else []
    problematic_phrases = llm_payload.get("problematic_phrases", []) if llm_payload else []
    explanation_points = llm_payload.get("explanations", []) if llm_payload else []
    rewrite_suggestion = clean_rewrite_candidate(llm_payload.get("rewrite", "")) if llm_payload else ""

    if not explanation_points:
        explanation_points = fallback_explanations(final_dimensions, context)
    if not problematic_phrases:
        problematic_phrases = detect_problematic_phrases(web_input_text, cue_analysis)
    if not special_situations:
        special_situations = detect_special_situations(final_dimensions, web_input_text, context, cue_analysis)
    llm_rewrite = clean_rewrite_candidate(llm_payload.get("rewrite", "")) if llm_payload else ""
    if is_acceptable_rewrite_candidate(llm_rewrite, cue_analysis):
        recommended_context = llm_rewrite
    else:
        recommended_context = build_recommended_context(web_input_text, final_dimensions, context, cue_analysis)
    rewrite_suggestion = recommended_context

    sentence_preview = [round(float(value), 4) for value in sentence_embedding[:8]]
    contextual_preview = [round(float(value), 4) for value in contextual_embedding[:8]]

    return PipelineOutput(
        context=context,
        recommended_context=recommended_context,
        audience_profile=audience_profile,
        original_text=web_input_text,
        normalized_text=normalized_text,
        tokens=tokens,
        embedding_model=embedder.model_name,
        sentence_embedding_preview=sentence_preview,
        contextual_embedding_preview=contextual_preview,
        heuristic_dimensions=heuristic_dimensions,
        llm_dimension_scores=llm_dimension_scores,
        final_dimensions=final_dimensions,
        validation_report=validation_report,
        heuristic_risk_score=heuristic_risk_score,
        llm_risk_score=llm_risk_score,
        final_risk_score=final_risk_score,
        risk_level=risk_level,
        threshold_type=active_threshold_type,
        issue_categories=issue_categories,
        special_situations=special_situations,
        problematic_phrases=problematic_phrases,
        explanation_points=explanation_points,
        llm_backend=llm_backend,
        llm_prompt=llm_prompt,
        rewrite_suggestion=rewrite_suggestion,
    )


def explain_scoring_ownership() -> str:
    return (
        "Human designers define the risk dimensions, rubrics, context multipliers, validation strategy, "
        "and score aggregation. The LLM estimates probability, severity, confidence, and rubric reasons "
        "under those rubrics. The final score is not pure LLM output; it is constrained by validation and "
        "weighted aggregation with the heuristic semantic scorer."
    )
