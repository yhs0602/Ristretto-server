# server.py
import time
import torch
import requests
import logging
import uvicorn
from io import BytesIO
from PIL import Image
from typing import List, Optional, Union, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel
from typing import List, Union, Literal, Optional
import base64
from PIL import Image

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


def load_image_from_url(url: str) -> torch.Tensor:
    if url.startswith("data:image"):
        # Base64 데이터 처리
        header, base64_data = url.split(",", 1)
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data)).convert("RGB")
    else:
        # 일반 URL 다운로드
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

    # 여기부터 기존 전처리 유지
    input_size = 384
    transform = build_transform(input_size)
    blocks = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=10
    )
    pixel_values = torch.stack([transform(im) for im in blocks])
    return pixel_values.to(torch.bfloat16).to(DEVICE)


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


class InputText(BaseModel):
    type: Literal["input_text"]
    text: str


class InputImage(BaseModel):
    type: Literal["input_image"]
    image_url: str


InputContent = Union[InputText, InputImage]


class ResponsesInput(BaseModel):
    role: Literal["user", "assistant"]
    content: List[InputContent]


class ResponsesRequest(BaseModel):
    model: str
    input: Union[ResponsesInput, List[ResponsesInput]]
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    # (생략) include, metadata, parallel_tool_calls, etc.


class ResponseMessage(BaseModel):
    type: Literal["message"]
    role: Literal["assistant"]
    content: List[dict]


class ResponsesResponse(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    status: str = "completed"
    output: List[ResponseMessage]
    model: str
    usage: dict


# ─── 핸들러 추가 ────────────────────────────────────────────────────────────
@app.post("/v1/responses", response_model=ResponsesResponse)
async def create_response(req: ResponsesRequest):
    if req.model != MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"Model {req.model} not available")

    # input이 ResponsesInput 단일일 수도 있고 리스트일 수도 있음
    inputs = req.input if isinstance(req.input, list) else [req.input]

    vision_inputs = []
    prompt_parts = []

    for message in inputs:
        role = message.role
        for content_item in message.content:
            if content_item.type == "input_text":
                prompt_parts.append(content_item.text)
            elif content_item.type == "input_image":
                # 여러 이미지가 들어올 수 있으므로 append
                img_tensor = load_image_from_url(content_item.image_url)
                vision_inputs.append(img_tensor)
                prompt_parts.append("<image>")

    # vision_inputs: 여러 이미지가 있을 경우 batch 처리용으로 stack
    if vision_inputs:
        vision_tensor = torch.cat(vision_inputs, dim=0)
    else:
        vision_tensor = None

    prompt = "\n".join(prompt_parts)

    gen_cfg = {
        "max_new_tokens": req.max_output_tokens or 2048,
        "temperature": req.temperature,
        "top_p": req.top_p,
    }

    # 모델 호출
    text, _ = model.chat(
        tokenizer,
        vision_tensor,
        prompt,
        gen_cfg,
        history=None,
        return_history=True,
    )

    msg = ResponseMessage(
        type="message",
        role="assistant",
        content=[{"type": "output_text", "text": text, "annotations": []}],
    )

    usage = {
        "input_tokens": 0,  # TODO: tokenizer로 계산 가능
        "output_tokens": 0,
        "total_tokens": 0,
    }

    return ResponsesResponse(
        id=f"resp_{int(time.time())}",
        created_at=int(time.time()),
        output=[msg],
        model=req.model,
        usage=usage,
    )


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
