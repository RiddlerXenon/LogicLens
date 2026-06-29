"""
LogicLens FastAPI Service
Классификатор логических ошибок + объяснение на русском через Qwen3-8B
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Пути к моделям ────────────────────────────────────────────────────────────
CLASSIFIER_PATH = "/data/logiclens/classifier-ru"
LLM_PATH        = "/data/models/qwen3-8b"

# ── Описания ошибок для промпта ───────────────────────────────────────────────
FALLACY_DESCRIPTIONS = {
    "ad hominem":            "атака на личность (переход на личности вместо обсуждения аргумента)",
    "ad populum":            "апелляция к большинству (аргумент «все так думают»)",
    "appeal to emotion":     "апелляция к эмоциям (манипуляция чувствами вместо логики)",
    "circular reasoning":    "круговое доказательство (тезис обосновывается самим собой)",
    "equivocation":          "эквивокация (подмена понятий, двусмысленное использование слов)",
    "fallacy of credibility":"апелляция к авторитету (некритичная ссылка на источник)",
    "fallacy of extension":  "расширение аргумента (приписывание оппоненту того, чего он не говорил)",
    "fallacy of logic":      "логическая ошибка (нарушение формальной логики)",
    "fallacy of relevance":  "ошибка релевантности (аргумент не относится к делу)",
    "false causality":       "ложная причинность (путаница между корреляцией и причиной)",
    "false dilemma":         "ложная дилемма (искусственное сужение вариантов до двух)",
    "faulty generalization": "поспешное обобщение (вывод по единичным случаям)",
    "intentional":           "намеренное введение в заблуждение (манипулятивная риторика)",
}

# ── Хранилище моделей ─────────────────────────────────────────────────────────
models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем модели при старте, освобождаем при остановке."""
    logger.info("Загрузка классификатора: %s", CLASSIFIER_PATH)
    tokenizer_cls = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
    model_cls = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH)
    device = 0 if torch.cuda.is_available() else -1
    model_cls = model_cls.to("cuda" if device == 0 else "cpu")
    model_cls.eval()
    models["tokenizer"] = tokenizer_cls
    models["classifier"] = model_cls
    models["device"] = device
    logger.info("Классификатор загружен (device=%s)", "cuda" if device == 0 else "cpu")

    logger.info("Загрузка Qwen3-8B: %s", LLM_PATH)
    models["llm"] = pipeline(
        "text-generation",
        model=LLM_PATH,
        device=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        max_new_tokens=512,
    )
    logger.info("Qwen3-8B загружен")

    yield

    models.clear()
    logger.info("Модели выгружены")


app = FastAPI(
    title="LogicLens API",
    description="Автоматическое выявление логических ошибок в тексте",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://10.19.84.130:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Схемы ─────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=2000, description="Текст для анализа")
    top_k: int = Field(3, ge=1, le=13, description="Сколько классов показать в scores")


class FallacyScore(BaseModel):
    label: str
    score: float


class AnalyzeResponse(BaseModel):
    text: str
    fallacy_type: str
    fallacy_type_ru: str
    confidence: float
    scores: list[FallacyScore]
    explanation: str


class HealthResponse(BaseModel):
    status: str
    classifier: bool
    llm: bool
    device: str


# ── Вспомогательные функции ───────────────────────────────────────────────────
def classify(text: str, top_k: int) -> tuple[str, float, list[FallacyScore]]:
    """Запускает классификатор, возвращает топ-класс + top_k scores."""
    tokenizer = models["tokenizer"]
    model     = models["classifier"]
    device    = "cuda" if models["device"] == 0 else "cpu"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    id2label = model.config.id2label

    top_indices = np.argsort(probs)[::-1]
    best_label  = id2label[int(top_indices[0])]
    best_score  = float(probs[top_indices[0]])

    scores = [
        FallacyScore(label=id2label[int(i)], score=round(float(probs[i]), 4))
        for i in top_indices[:top_k]
    ]
    return best_label, best_score, scores


def build_prompt(text: str, fallacy: str) -> str:
    description = FALLACY_DESCRIPTIONS.get(fallacy, fallacy)
    llm_tok = models["llm"].tokenizer

    system = (
        "Ты эксперт по логике и критическому мышлению. "
        "Отвечай только самим объяснением — без вводных фраз, без «пример ответа», без повторений."
    )
    user = (
        f'Утверждение: «{text}»\n\n'
        f'Тип логической ошибки: {fallacy} — {description}.\n\n'
        f'Напиши связный текст из 5–7 предложений без нумерации и заголовков.\n'
        f'Сначала объясни, в чём конкретно состоит ошибка в этом тексте и почему это логически некорректно.\n'
        f'Затем приведи один реальный пример похожей ошибки из жизни или истории.\n'
        f'В конце покажи, как правильно переформулировать исходный аргумент.'
    )

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        # enable_thinking=False отключает режим reasoning у Qwen3
        return llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


_STOP_SEQUENCES = [
    "\n\nТакой формат",
    "\n\nДа, такой",
    "\n\nЯ готов",
    "\n\nХорошо,",
    "\n\nКонечно!",
    "\n\nПример ответа",
    "\n\nВот пример",
    "\n\n---",
    "<|im_end|>",
    "<|endoftext|>",
    "<|im_start|>",
]


def _trim_explanation(text: str) -> str:
    """Убирает <think> блок Qwen3 и обрезает хвост после стоп-маркеров."""
    import re
    # Убираем thinking блок целиком (включая содержимое)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # Обрезаем хвост
    for marker in _STOP_SEQUENCES:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def generate_explanation(text: str, fallacy: str) -> str:
    """Генерирует объяснение через Qwen3-8B."""
    tokenizer = models["llm"].tokenizer
    prompt = build_prompt(text, fallacy)

    stop_ids = []
    for seq in _STOP_SEQUENCES:
        ids = tokenizer.encode(seq, add_special_tokens=False)
        if ids:
            stop_ids.append(ids[0])  # первый токен как мягкий стоп

    result = models["llm"](
        prompt,
        do_sample=False,
        temperature=None,
        top_p=None,
        repetition_penalty=1.15,
        max_new_tokens=600,
        eos_token_id=[tokenizer.eos_token_id] + stop_ids,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    raw = result[0]["generated_text"].strip()
    return _trim_explanation(raw)


# ── Эндпоинты ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return HealthResponse(
        status="ok",
        classifier="classifier" in models,
        llm="llm" in models,
        device="cuda" if models.get("device") == 0 else "cpu",
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(req: AnalyzeRequest):
    """
    Принимает текст → определяет тип логической ошибки → генерирует объяснение.
    """
    if "classifier" not in models or "llm" not in models:
        raise HTTPException(status_code=503, detail="Модели ещё загружаются, попробуйте позже")

    # 1. Классификация
    fallacy_en, confidence, scores = classify(req.text, req.top_k)

    # 2. Генерация объяснения
    explanation = generate_explanation(req.text, fallacy_en)

    return AnalyzeResponse(
        text=req.text,
        fallacy_type=fallacy_en,
        fallacy_type_ru=FALLACY_DESCRIPTIONS.get(fallacy_en, fallacy_en),
        confidence=round(confidence, 4),
        scores=scores,
        explanation=explanation,
    )


# ── Debate mode ───────────────────────────────────────────────────────────────

# Веса серьёзности ошибок (чем выше — тем хуже для аргумента)
FALLACY_SEVERITY = {
    "intentional":           1.0,
    "ad hominem":            0.9,
    "appeal to emotion":     0.8,
    "false causality":       0.8,
    "false dilemma":         0.75,
    "circular reasoning":    0.75,
    "equivocation":          0.7,
    "fallacy of extension":  0.7,
    "ad populum":            0.65,
    "fallacy of credibility":0.6,
    "fallacy of relevance":  0.6,
    "faulty generalization": 0.55,
    "fallacy of logic":      0.5,
}


class DebateRequest(BaseModel):
    topic: str  = Field("", max_length=300, description="Тема дебатов (необязательно)")
    text_a: str = Field(..., min_length=5, max_length=2000, description="Аргумент участника A")
    text_b: str = Field(..., min_length=5, max_length=2000, description="Аргумент участника B")
    name_a: str = Field("Участник A", max_length=50)
    name_b: str = Field("Участник B", max_length=50)


class DebaterResult(BaseModel):
    name: str
    text: str
    fallacy_type: str
    fallacy_type_ru: str
    confidence: float
    severity: float          # 0–1, насколько серьёзна ошибка
    logic_score: float       # 0–100, итоговый балл логичности
    scores: list[FallacyScore]
    feedback: str            # краткий разбор от LLM


class DebateResponse(BaseModel):
    topic: str
    player_a: DebaterResult
    player_b: DebaterResult
    winner: str              # "A", "B" или "draw"
    winner_name: str
    verdict: str             # развёрнутый вердикт от LLM
    score_diff: float


def compute_logic_score(confidence: float, fallacy: str) -> float:
    """0–100: чем меньше уверенность классификатора и чем мягче ошибка, тем выше балл."""
    severity = FALLACY_SEVERITY.get(fallacy, 0.6)
    # penalty = насколько явно выражена ошибка
    penalty = confidence * severity * 100
    return round(max(0.0, 100.0 - penalty), 1)


def build_feedback_prompt(name: str, text: str, fallacy: str, confidence: float) -> str:
    description = FALLACY_DESCRIPTIONS.get(fallacy, fallacy)
    llm_tok = models["llm"].tokenizer
    system = (
        "Ты судья в дебатах. Оценивай аргументы кратко и по существу. "
        "Отвечай только самим разбором, без вводных фраз и повторений."
    )
    conf_pct = f"{confidence * 100:.0f}%"
    user = (
        f'Участник «{name}» утверждает: «{text}»\n\n'
        f'Классификатор обнаружил логическую ошибку «{fallacy}» ({description}) '
        f'с уверенностью {conf_pct}.\n\n'
        f'Напиши 2–3 предложения: в чём слабость этого аргумента и что можно улучшить.'
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def build_verdict_prompt(
    topic: str,
    name_a: str, text_a: str, fallacy_a: str, score_a: float,
    name_b: str, text_b: str, fallacy_b: str, score_b: float,
) -> str:
    llm_tok = models["llm"].tokenizer
    desc_a = FALLACY_DESCRIPTIONS.get(fallacy_a, fallacy_a)
    desc_b = FALLACY_DESCRIPTIONS.get(fallacy_b, fallacy_b)
    winner_name = name_a if score_a > score_b else (name_b if score_b > score_a else "ничья")

    system = (
        "Ты беспристрастный судья логических дебатов. "
        "Выноси вердикт чётко и обоснованно, без вводных фраз."
    )
    topic_line = f'Тема дебатов: «{topic}»\n\n' if topic.strip() else ""
    user = (
        f'{topic_line}'
        f'«{name_a}»: «{text_a}» — ошибка: {fallacy_a} ({desc_a}), балл логичности: {score_a}/100\n'
        f'«{name_b}»: «{text_b}» — ошибка: {fallacy_b} ({desc_b}), балл логичности: {score_b}/100\n\n'
        f'Победитель по баллам: {winner_name}.\n\n'
        f'Напиши вердикт: 4–5 предложений связного текста без нумерации. '
        f'Объясни, чей аргумент логически сильнее и почему, '
        f'укажи главные слабости каждого участника, '
        f'и дай один совет каждому для улучшения аргументации.'
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def llm_generate(prompt: str, max_new_tokens: int = 400) -> str:
    tokenizer = models["llm"].tokenizer
    stop_ids = []
    for seq in _STOP_SEQUENCES:
        ids = tokenizer.encode(seq, add_special_tokens=False)
        if ids:
            stop_ids.append(ids[0])
    result = models["llm"](
        prompt,
        do_sample=False,
        temperature=None,
        top_p=None,
        repetition_penalty=1.15,
        max_new_tokens=max_new_tokens,
        eos_token_id=[tokenizer.eos_token_id] + stop_ids,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    return _trim_explanation(result[0]["generated_text"].strip())


@app.post("/debate", response_model=DebateResponse, tags=["debate"])
def debate(req: DebateRequest):
    """
    Дебаты: два аргумента → классификация + оценка каждого → вердикт победителя.
    """
    if "classifier" not in models or "llm" not in models:
        raise HTTPException(status_code=503, detail="Модели ещё загружаются")

    # Классификация обоих аргументов
    fallacy_a, conf_a, scores_a = classify(req.text_a, 3)
    fallacy_b, conf_b, scores_b = classify(req.text_b, 3)

    sev_a = FALLACY_SEVERITY.get(fallacy_a, 0.6)
    sev_b = FALLACY_SEVERITY.get(fallacy_b, 0.6)
    score_a = compute_logic_score(conf_a, fallacy_a)
    score_b = compute_logic_score(conf_b, fallacy_b)

    # Краткий фидбек каждому участнику
    feedback_a = llm_generate(build_feedback_prompt(req.name_a, req.text_a, fallacy_a, conf_a), 250)
    feedback_b = llm_generate(build_feedback_prompt(req.name_b, req.text_b, fallacy_b, conf_b), 250)

    # Общий вердикт
    verdict = llm_generate(
        build_verdict_prompt(
            req.topic,
            req.name_a, req.text_a, fallacy_a, score_a,
            req.name_b, req.text_b, fallacy_b, score_b,
        ),
        500,
    )

    # Победитель
    if score_a > score_b + 2:
        winner, winner_name = "A", req.name_a
    elif score_b > score_a + 2:
        winner, winner_name = "B", req.name_b
    else:
        winner, winner_name = "draw", "Ничья"

    return DebateResponse(
        topic=req.topic,
        player_a=DebaterResult(
            name=req.name_a, text=req.text_a,
            fallacy_type=fallacy_a, fallacy_type_ru=FALLACY_DESCRIPTIONS.get(fallacy_a, fallacy_a),
            confidence=round(conf_a, 4), severity=sev_a,
            logic_score=score_a, scores=scores_a, feedback=feedback_a,
        ),
        player_b=DebaterResult(
            name=req.name_b, text=req.text_b,
            fallacy_type=fallacy_b, fallacy_type_ru=FALLACY_DESCRIPTIONS.get(fallacy_b, fallacy_b),
            confidence=round(conf_b, 4), severity=sev_b,
            logic_score=score_b, scores=scores_b, feedback=feedback_b,
        ),
        winner=winner,
        winner_name=winner_name,
        verdict=verdict,
        score_diff=round(abs(score_a - score_b), 1),
    )
