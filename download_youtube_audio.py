#!/usr/bin/env python3
"""
Скачивает аудио с YouTube по ссылке и сохраняет в ./audio_samples.

Зависимости:
  - yt-dlp
  - ffmpeg

Пример:
  python3 download_youtube_audio.py "https://www.youtube.com/watch?v=4XqwA8WxEn8"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _find_ytdlp_runner() -> list[str] | None:
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    try:
        import yt_dlp  # noqa: F401
    except ModuleNotFoundError:
        return None
    return [sys.executable, "-m", "yt_dlp"]


def main(argv: list[str]) -> int:
    repo_dir = Path(__file__).resolve().parent
    default_out_dir = repo_dir / "audio_samples"

    parser = argparse.ArgumentParser(
        description="Скачает аудио дорожку с YouTube и сохранит в папку audio_samples."
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="Ссылка на YouTube (если не указать — скрипт попросит вставить ссылку)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(default_out_dir),
        help=f"Папка для сохранения (по умолчанию: {default_out_dir})",
    )
    parser.add_argument(
        "--format",
        default="wav",
        choices=("wav", "mp3", "m4a", "opus"),
        help="Формат аудио на выходе (по умолчанию: wav)",
    )
    args = parser.parse_args(argv)

    url = (args.url or "").strip()
    if not url:
        try:
            url = input("Вставь ссылку на YouTube: ").strip()
        except EOFError:
            url = ""
    if not url:
        parser.error("Нужна ссылка: передай url аргументом или вставь при запросе")

    ytdlp = _find_ytdlp_runner()
    if not ytdlp:
        print(
            "Не найден yt-dlp.\n"
            "Установи один из вариантов:\n"
            "  python3 -m pip install -U yt-dlp\n"
            "  brew install yt-dlp",
            file=sys.stderr,
        )
        return 2

    if not shutil.which("ffmpeg"):
        print(
            "Не найден ffmpeg (нужен для извлечения/конвертации аудио).\n"
            "Установка (macOS): brew install ffmpeg",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_template = str(out_dir / "%(title).200B_%(id)s.%(ext)s")

    cmd = [
        *ytdlp,
        "--no-playlist",
        "--restrict-filenames",
        "-x",
        "--audio-format",
        args.format,
        "--audio-quality",
        "0",
        "-o",
        out_template,
        url,
    ]

    print("Запуск:", " ".join(cmd))
    result = subprocess.run(cmd)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
