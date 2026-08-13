#!/usr/bin/env python3
"""
Заливка сконвертированных ONNX-весов GigaAM на Hugging Face Hub.

Запуск в контейнере (веса в томе) с токеном в env:
    docker exec gigaam-stt pip install --user huggingface_hub
    docker exec -e HF_TOKEN=... gigaam-stt python /tmp/upload_onnx_hf.py \
        --repo vvpreo/gigaam-v3-onnx --dir /app/data/onnx

Токен нужен с правом Write. Репозиторий создаётся, если его нет.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

README = """---
license: mit
language: [ru]
tags: [asr, speech-recognition, gigaam, onnx, russian]
---

# GigaAM v3 — ONNX weights (+ int8) + GigaAMEmo

ONNX-конвертация моделей **GigaAM v3** (SberDevices, MIT) для инференса на
ONNX Runtime без PyTorch, плюс модель эмоций **GigaAMEmo**.

Сконвертировано из официальных чекпойнтов
(`cdn.chatwm.opensmodel.sberdevices.ru/GigaAM`) скриптом
[`scripts/convert_onnx.py`](https://github.com/vvpreo/gigaam-stt) проекта
gigaam-stt (vendored `gigaam.to_onnx()` + `onnxruntime.quantization.quantize_dynamic`).
Раскладка файлов совместима с `gigaam.onnx_utils.load_onnx`.

| Модель | Файлы | Назначение |
|---|---|---|
| `v3_e2e_ctc` | `.onnx` / `.int8.onnx` + `.yaml` | ASR, пунктуация+нормализация, самый быстрый |
| `v3_e2e_rnnt` | `_encoder/_decoder/_joint` (+int8) + `.yaml` | ASR, лучшее качество |
| `v3_ctc`, `v3_rnnt` | аналогично | ASR без пунктуации |
| `emo` | `.onnx` / `.int8.onnx` + `.yaml` | эмоции: angry/sad/neutral/positive |

Токенизаторы (sentencepiece, для `*_e2e_*`) скачиваются с CDN SberDevices:
`{name}_tokenizer.model`.

Используется сервисом [gigaam-stt](https://hub.docker.com/) — русский STT
с OpenAI-совместимым API в одном Docker-образе.

Лицензия — MIT (наследуется от GigaAM, SberDevices).
"""

LICENSE_NOTE = """MIT License

Derived from GigaAM (Copyright SberDevices, MIT License):
https://github.com/salute-developers/GigaAM

Conversion artifacts (ONNX export + int8 quantization) are distributed
under the same MIT terms.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="vvpreo/gigaam-v3-onnx")
    parser.add_argument("--dir", default="/app/data/onnx")
    args = parser.parse_args()

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True)

    src = Path(args.dir)
    api.upload_file(
        path_or_fileobj=README.encode(), path_in_repo="README.md",
        repo_id=args.repo,
    )
    api.upload_file(
        path_or_fileobj=LICENSE_NOTE.encode(), path_in_repo="LICENSE",
        repo_id=args.repo,
    )
    api.upload_folder(
        folder_path=str(src), repo_id=args.repo,
        allow_patterns=["*.onnx", "*.yaml"],
    )
    print(f"Uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
