#!/usr/bin/env python3
"""Expand a short R2V prompt into a first-frame-anchored R2V prompt.

Reads every record (ref_images + a short text), shows the reference images and the
short description to a local multimodal model (thinking off), and writes the rewritten,
detailed prompt back into that record. Resumable: a record whose output field already
has a non-empty value is skipped, so you can Ctrl+C and re-run anytime.

    python r2v_pe.py                          # run / resume on the default metadata.jsonl
    python r2v_pe.py --limit 3                # quick check on a few records first
    python r2v_pe.py --in-field caption       # read the short text from a different field
"""
import argparse
import base64
import json
import mimetypes
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from openai import OpenAI

RI2V_PROMPT_ENHANCE_TEMPLATE = """You are an expert at writing reference-image-to-video prompts for a model where image0 is the exact first frame.

I'm providing you with:
1. {image_num} reference image(s), referred to as image0, image1, image2, ... in order. image0 is the exact first frame of the generated video. image1, image2, etc. are identity/detail references only.
2. An original video description text.

Rewrite the original description as ONE polished English prompt, not two sections. The prompt must follow this exact structure:

1. Start with this sentence pattern:
   "Use image0 as the exact first frame and preserve its [portrait/landscape] composition, camera angle, subject identity, [initial pose], clothing, lighting, and background."
2. Then describe the first frame in concrete visual detail, strictly from image0:
   - subject type, age/gender if visible, pose, gaze direction, facial expression;
   - clothing, accessories, handheld objects, body/hand positions;
   - spatial layout, background objects, lighting, and atmosphere.
3. Then describe the video action as a clear temporal sequence:
   "In the video, ..." followed by "first...", "After a brief pause...", "Then...", "Once...", as needed.
   The action sequence must come from the original description. If the original text says the subject turns head, looks at the camera, waves, transfers an object, stands up, raises a V-sign, etc., describe those steps exactly and naturally.
4. End with a stability sentence:
   "Keep the camera static, preserve the same [portrait/landscape] framing, and maintain stable [identity/clothing/important objects/background/lighting] throughout the motion."

Hard requirements:
- image0 is the first frame anchor. The first-frame description MUST be based on image0 only.
- image1, image2, etc. may be used only to refine identity details such as facial features, hairstyle, or expression. They MUST NOT change the scene, outfit, pose, or starting composition from image0.
- Do NOT invent images that are not provided. If {image_num} images are provided, valid references are only image0 through image{last_image_index}.
- Do NOT mention "image4" or any higher index unless it is actually provided.
- Do NOT add actions that are not in the original description. Do not add waving, standing up, camera orbit, push-in, or walking unless the original description asks for it.
- Prefer a static camera. Only allow a very subtle camera shift if the original description explicitly asks for camera movement. Never add cinematic orbit/push-in by default.
- Keep object hand ownership accurate. If image0 shows an object in a hand and the original description asks to transfer it, describe the transfer step explicitly.
- The output must be entirely in English.
- Return ONLY a JSON object with one key: "rewritten_text". The value should be the final prompt string. No markdown, no notes, no explanations.

Style target:
Use the same style as this example:
"Use image0 as the exact first frame and preserve its portrait composition, camera angle, subject identity, standing pose, clothing, lighting, and background. The first frame shows ... In the video, the subject first ... After a brief pause, ... Then ... Keep the camera static, preserve the same portrait framing, and maintain stable facial identity, clothing details, important objects, background, and lighting throughout the motion."

Original description:
{original_text}
"""

_lock = threading.Lock()


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def parse_caption(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S).strip()
    for cand in (s, *(m.group(0) for m in re.finditer(r"\{.*?\}", s, flags=re.S))):
        try:
            cap = json.loads(cand).get("rewritten_text")
            if isinstance(cap, str) and cap.strip():
                return cap.strip()
        except Exception:
            continue
    if s and not s.lstrip().startswith("{"):
        return s
    raise ValueError(f"no rewritten_text in reply: {text!r:.200}")


def validate_caption(text: str, image_num: int) -> None:
    if not text.startswith("Use image0 as the exact first frame"):
        raise ValueError("rewritten prompt does not start with the required first-frame anchor")
    if "In the video," not in text:
        raise ValueError("rewritten prompt is missing the video action sequence")
    if "Keep the camera static" not in text:
        raise ValueError("rewritten prompt is missing the camera/static stability sentence")
    bad_refs = sorted(
        {int(m.group(1)) for m in re.finditer(r"\bimage(\d+)\b", text)}
        - set(range(image_num))
    )
    if bad_refs:
        raise ValueError(f"rewritten prompt references missing image indexes: {bad_refs}")
    if re.search(r"\bPart\s*[12]\b|Short instruction|Long instruction|Generate a video where", text, re.I):
        raise ValueError("rewritten prompt still uses the old two-part format")


def enhance_one(rec, root, client, model, in_field):
    original = (rec.get(in_field) or "").strip()
    if not original:
        raise ValueError(f"empty input field {in_field!r}")
    refs = [(root / p).resolve() for p in rec["ref_images"]]
    for p in refs:
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")
    content = [{"type": "text", "text": "Reference images (image0, image1, ...). "
                                        "Rewrite the R2V prompt now."}]
    for img in refs:
        content.append({"type": "image_url", "image_url": {"url": data_uri(img)}})
    system = RI2V_PROMPT_ENHANCE_TEMPLATE.format(
        image_num=len(refs),
        last_image_index=len(refs) - 1,
        original_text=original,
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": content}]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=2048, temperature=0.4,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            cap = parse_caption(resp.choices[0].message.content)
            if len(cap) < 40:
                raise ValueError(f"too short: {cap!r}")
            validate_caption(cap, len(refs))
            return cap
        except Exception as e:  # noqa: BLE001 - retry transient API/IO/parse errors
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="/datasets/codes_zsqiao/train_r2v_v0.2/metadata.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--base-url", default="http://localhost:30000/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--in-field", default="short_prompt", help="record field holding the short prompt")
    ap.add_argument("--out-field", default="prompt", help="record field to write the rewritten prompt into")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N pending records")
    args = ap.parse_args()

    meta = Path(args.metadata).resolve()
    root = meta.parent
    records = [json.loads(l) for l in meta.read_text().splitlines() if l.strip()]

    def save():  # atomic rewrite of the whole file
        tmp = meta.with_suffix(meta.suffix + ".tmp")
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
        tmp.replace(meta)

    pending = [r for r in records
               if (r.get(args.in_field) or "").strip() and not (r.get(args.out_field) or "").strip()]
    if args.limit:
        pending = pending[:args.limit]
    print(f"{len(records)} records, {len(pending)} still to do", flush=True)
    if not pending:
        return

    client = OpenAI(base_url=args.base_url, api_key="EMPTY",
                    http_client=httpx.Client(trust_env=False, timeout=600))
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(enhance_one, r, root, client, args.model, args.in_field): r for r in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = futs[fut]
            try:
                rec[args.out_field] = fut.result()
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"[{i}/{len(pending)}] FAIL {rec.get('video', rec.get(args.in_field))}: {e}", flush=True)
                continue
            with _lock:
                save()
            ok += 1
            if i % 20 == 0 or i == len(pending):
                print(f"[{i}/{len(pending)}] ok={ok} fail={fail}", flush=True)
    print(f"done: ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
