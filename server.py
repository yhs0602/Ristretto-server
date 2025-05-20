# server.py
import time
import json
import torch
import requests
import logging
import uvicorn
from io import BytesIO
from PIL import Image
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ─── 로깅 설정 ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vlm-server")

# ─── 모델 로딩 ─────────────────────────────────────────────────────────
MODEL_NAME = "LiAutoAD/Ristretto-3B"
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# 전처리 파이프라인
IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for r in target_ratios:
        ar = r[0] / r[1]
        diff = abs(aspect_ratio - ar)
        if diff < best_diff or (
            diff == best_diff and area > 0.5 * image_size * image_size * r[0] * r[1]
        ):
            best_diff, best = diff, r
    return best


def dynamic_preprocess(
    image: Image.Image, min_num=1, max_num=10, image_size=448, use_thumbnail=False
):
    w, h = image.size
    ar = w / h
    # 가능한 종횡비 조합
    ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    tgt = find_closest_aspect_ratio(ar, ratios, w, h, image_size)
    tw, th = image_size * tgt[0], image_size * tgt[1]
    resized = image.resize((tw, th))
    blocks = []
    for i in range(tgt[0] * tgt[1]):
        cols = tw // image_size
        x0 = (i % cols) * image_size
        y0 = (i // cols) * image_size
        blocks.append(resized.crop((x0, y0, x0 + image_size, y0 + image_size)))
    if use_thumbnail and len(blocks) > 1:
        blocks.append(image.resize((image_size, image_size)))
    return blocks


def load_image_from_url(url: str, input_size=384, max_num=10):
    resp = requests.get(url, timeout=10)
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    imgs = dynamic_preprocess(
        img, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    tf = build_transform(input_size)
    pix = torch.stack([tf(im) for im in imgs])
    return pix.to(torch.bfloat16).to(DEVICE)


logger.info("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, use_fast=False
)
model = (
    AutoModel.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    .eval()
    .to(DEVICE)
)
logger.info("Model loaded onto %s", DEVICE)

# ─── FastAPI 앱 & 스키마 ─────────────────────────────────────────────────
app = FastAPI(title="Ristretto-3B VLM (OpenAI-compatible)", version="1.0.0")


class ChatMessage(BaseModel):
    role: str
    content: Union[str, Dict[str, str]]  # str 또는 {"image_url": "..."} 형태


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False


class Choice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    if req.model != MODEL_NAME:
        raise HTTPException(400, f"Unknown model: {req.model}")

    # 메시지 → model.chat 입력 변환
    vision_inputs = None
    texts = []
    for msg in req.messages:
        if isinstance(msg.content, dict) and "image_url" in msg.content:
            vision_inputs = load_image_from_url(msg.content["image_url"])
            texts.append("<image>")
        else:
            texts.append(
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )

    prompt = "".join(f"{m}\n" for m in texts)
    gen_cfg = {
        "max_new_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
    }
    # 모델 추론
    response, _history = model.chat(
        tokenizer, vision_inputs, prompt, gen_cfg, history=None, return_history=True
    )
    # 토큰 수 계산
    # (simplified: prompt length + response length)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    response_ids = tokenizer(response, return_tensors="pt").input_ids[0]
    usage = Usage(
        prompt_tokens=prompt_ids.size(0),
        completion_tokens=response_ids.size(0),
        total_tokens=prompt_ids.size(0) + response_ids.size(0),
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        object="chat.completion",
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message={"role": "assistant", "content": response},
                finish_reason="stop",
            )
        ],
        usage=usage,
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
