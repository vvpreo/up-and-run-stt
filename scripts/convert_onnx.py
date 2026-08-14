#!/usr/bin/env python3
"""
Конвертация весов GigaAM (.ckpt с CDN Сбера) в ONNX + int8-квантизация.

Требует PyTorch + onnx + onnxruntime, поэтому запускается в контейнере
(или окружении) с torch — публичный ONNX-образ сервиса конвертировать
не умеет и скачивает уже готовые файлы:

    docker exec up-and-run-stt pip install --user onnx onnxruntime
    docker exec up-and-run-stt python /tmp/convert_onnx.py \
        --out /app/data/onnx --cache /app/data/gigaam

Результат — раскладка, совместимая с vendored `gigaam.onnx_utils.load_onnx`
и с комьюнити-репозиторием istupakov/gigaam-v3-onnx:
    {name}.onnx / {name}_encoder|_decoder|_joint.onnx (RNNT) + {name}.yaml
    + {name}.int8.onnx (квантизованные варианты)
"""

import argparse
from pathlib import Path

import gigaam
from onnxruntime.quantization import QuantType, quantize_dynamic

ASR_VARIANTS = ["v3_e2e_ctc", "v3_ctc", "v3_e2e_rnnt", "v3_rnnt"]


def convert(name: str, out_dir: str, cache_dir: str) -> None:
    print(f"=== {name}: loading (weights download on first use) ===", flush=True)
    model = gigaam.load_model(
        name,
        fp16_encoder=False,
        use_flash=False,
        device="cpu",
        download_root=cache_dir,
    )
    print(f"=== {name}: exporting to ONNX ===", flush=True)
    model.to_onnx(out_dir)
    del model

    # int8-квантизация каждого экспортированного файла варианта
    for f in sorted(Path(out_dir).glob(f"{name}*.onnx")):
        if ".int8." in f.name:
            continue
        target = f.with_name(f.name.replace(".onnx", ".int8.onnx"))
        if target.exists():
            continue
        print(f"=== quantizing {f.name} -> {target.name} ===", flush=True)
        quantize_dynamic(str(f), str(target), weight_type=QuantType.QInt8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/app/data/onnx")
    parser.add_argument("--cache", default="/app/data/gigaam")
    parser.add_argument(
        "--variants", default=",".join(ASR_VARIANTS + ["emo"]),
        help="через запятую; по умолчанию все ASR + emo",
    )
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        convert(name, args.out, args.cache)

    print("=== DONE ===", flush=True)
    for f in sorted(Path(args.out).iterdir()):
        print(f"  {f.name}  {f.stat().st_size >> 20} MB", flush=True)


if __name__ == "__main__":
    main()
