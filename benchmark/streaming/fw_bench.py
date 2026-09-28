#!/usr/bin/env python3
"""
Замер faster-whisper (CTranslate2, CUDA) для псевдо-стриминга: стоимость окна
10 с одиночно и в батче из окон РАЗНЫХ пользователей (3 и 6), greedy и beam 5,
fp16 и int8_float16; плюс полный клип через BatchedInferencePipeline (RTF, текст).

Батч разных окон собирается напрямую через ctranslate2: фичи каждого окна ->
stack -> model.encode -> model.generate с одинаковым промптом на каждую
строку — ровно то, что делает BatchedInferencePipeline внутри для сегментов
одного файла, но здесь окна из разных мест клипа, как от разных сессий.

  python3 fw_bench.py --audio /audio/x.wav --model large-v3-turbo --out /out/fw.json \
      [--compute float16,int8_float16] [--batches 1,3,6] [--runs 5] [--skip-full]
"""

import argparse
import json
import statistics
import sys
import time
import wave

import numpy as np


def mem_avail_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return -1.0


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def pct(v, p):
    s = sorted(v)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 4)


def summary(v):
    return {"n": len(v), "mean": round(statistics.fmean(v), 4), "p50": pct(v, 0.5), "max": round(max(v), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--download-root", default="/models")
    ap.add_argument("--compute", default="float16,int8_float16")
    ap.add_argument("--batches", default="1,3,6")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--window-sec", type=float, default=10.0)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ctranslate2
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    audio = read_wav(args.audio)
    sr = 16000
    win = int(args.window_sec * sr)
    # окна из разных мест клипа — как от разных пользователей
    offsets = [0, 20, 40, 60, 80, 100]  # 0–10 с: «Проверяем транскрипцию через GigaChat…»
    windows = [audio[int(o * sr): int(o * sr) + win] for o in offsets]
    assert all(len(w) == win for w in windows), "клип короче, чем нужно для 6 окон"

    report = {"model": args.model, "ct2": ctranslate2.__version__, "window_sec": args.window_sec,
              "compute_types": sorted(ctranslate2.get_supported_compute_types("cuda")), "results": {}}
    print("supported compute types on cuda:", report["compute_types"], file=sys.stderr)

    for ct in args.compute.split(","):
        m0 = mem_avail_gib()
        t0 = time.perf_counter()
        model = WhisperModel(args.model, device="cuda", compute_type=ct, download_root=args.download_root)
        load_sec = time.perf_counter() - t0
        res = {"load_sec": round(load_sec, 2)}

        from faster_whisper.tokenizer import Tokenizer
        tokenizer = Tokenizer(model.hf_tokenizer, True, task="transcribe", language="ru")
        prompt = model.get_prompt(tokenizer, previous_tokens=[], without_timestamps=True, prefix=None)
        nb_max = model.feature_extractor.nb_max_frames

        def feats(w):
            # padding=True: экстрактор сам дополняет АУДИО тишиной до 30 с и считает
            # лог-мел; дополнять нулями уже готовые фичи нельзя — ноль в лог-мел
            # пространстве это не тишина, модель на таком «шуме» галлюцинирует.
            return model.feature_extractor(w, padding=True)[..., : nb_max]

        def run_batch(ws, beam):
            t = time.perf_counter()
            f = np.stack([feats(w) for w in ws]).astype(np.float32)
            t_feat = time.perf_counter() - t
            sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(f))
            enc = model.model.encode(sv, to_cpu=False)
            out = model.model.generate(enc, [prompt] * len(ws), beam_size=beam, max_length=224,
                                       suppress_blank=True, return_scores=False, sampling_temperature=0.0)
            texts = [tokenizer.decode(o.sequences_ids[0]).strip() for o in out]
            return time.perf_counter() - t, t_feat, texts

        # прогрев
        for _ in range(2):
            run_batch(windows[:1], 1)
        res["mem_after_load_gib"] = round(m0 - mem_avail_gib(), 2)

        for b in [int(x) for x in args.batches.split(",")]:
            for beam in ([1, 5] if b == 3 else [1]):
                totals, feat_t, texts = [], [], None
                for _ in range(args.runs):
                    tt, tf, texts = run_batch(windows[:b], beam)
                    totals.append(tt); feat_t.append(tf)
                key = f"batch{b}_beam{beam}"
                res[key] = {"total_sec": summary(totals), "features_sec": summary(feat_t),
                            "per_window_sec": round(statistics.fmean(totals) / b, 4),
                            "windows_per_sec": round(b / statistics.fmean(totals), 2),
                            "text0": texts[0][:120]}
                print(f"[{ct}] {key}: total p50 {res[key]['total_sec']['p50']} s, "
                      f"per window {res[key]['per_window_sec']} s, {res[key]['windows_per_sec']} win/s | {texts[0][:80]}",
                      file=sys.stderr)

        if not args.skip_full:
            pipe = BatchedInferencePipeline(model)
            t = time.perf_counter()
            segs, info = pipe.transcribe(audio, language="ru", beam_size=1, batch_size=8, vad_filter=True,
                                         initial_prompt="Диктовка на русском с английскими IT-терминами: GigaChat, GitHub, feature branch, pull request.",
                                         condition_on_previous_text=False, without_timestamps=True)
            text = " ".join(s.text.strip() for s in segs)
            full = time.perf_counter() - t
            res["full_clip"] = {"audio_sec": round(len(audio) / sr, 1), "wall_sec": round(full, 2),
                                "rtf": round(full / (len(audio) / sr), 4), "text_head": text[:300]}
            print(f"[{ct}] full clip {len(audio)/sr:.0f}s in {full:.2f}s (RTF {full/(len(audio)/sr):.4f}) | {text[:100]}", file=sys.stderr)

        report["results"][ct] = res
        del model
        import gc; gc.collect()

    js = json.dumps(report, ensure_ascii=False, indent=2)
    print(js)
    if args.out:
        open(args.out, "w").write(js)


if __name__ == "__main__":
    main()
