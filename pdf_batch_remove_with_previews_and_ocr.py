# =========================
# pdf_batch_remove_with_previews_and_ocr.py
# =========================
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from pdf2image import convert_from_path
from PIL import Image


# -------------------- Tipos/Config --------------------
@dataclass
class RGBThresholds:
    rmin: int
    rmax: int
    gmin: int
    gmax: int
    bmin: int
    bmax: int
    inclusive: bool

    def validate(self) -> None:
        for name, val in [
            ("rmin", self.rmin),
            ("rmax", self.rmax),
            ("gmin", self.gmin),
            ("gmax", self.gmax),
            ("bmin", self.bmin),
            ("bmax", self.bmax),
        ]:
            if not (0 <= val <= 255):
                raise ValueError(f"{name} deve estar em [0,255], recebido {val}")
        if not (
            self.rmin < self.rmax and self.gmin < self.gmax and self.bmin < self.bmax
        ):
            raise ValueError("Para cada canal, min < max.")


# -------------------- Regras / Máscara --------------------
def judge(x: int, y: int) -> bool:
    temp = -(600.0 / 1575.0) * x
    return 1350 + temp < y < 1500 + temp


def ensure_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    return arr


def build_mask(
    arr: np.ndarray, th: RGBThresholds, apply_judge_region: bool
) -> np.ndarray:
    arr = ensure_rgb(arr)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    if th.inclusive:
        mr = (r >= th.rmin) & (r <= th.rmax)
        mg = (g >= th.gmin) & (g <= th.gmax)
        mb = (b >= th.bmin) & (b <= th.bmax)
    else:
        mr = (r > th.rmin) & (r < th.rmax)
        mg = (g > th.gmin) & (g < th.gmax)
        mb = (b > th.bmin) & (b < th.bmax)
    mask_rgb = mr & mg & mb
    if not apply_judge_region:
        return mask_rgb
    h, w = r.shape
    yy, xx = np.mgrid[0:h, 0:w]
    region = (-(600.0 / 1575.0) * xx + 1350 < yy) & (yy < -(600.0 / 1575.0) * xx + 1500)
    return mask_rgb & region


# -------------------- Saída principal --------------------
def handle_remove_to_white(imgs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    imgs = ensure_rgb(imgs)
    out = imgs.copy()
    out[mask] = 255
    return out


# -------------------- Previews opcionais --------------------
def make_only_removed(arr: np.ndarray, mask: np.ndarray, bg_value: int) -> np.ndarray:
    arr = ensure_rgb(arr)
    out = np.full_like(arr, fill_value=bg_value, dtype=np.uint8)
    out[mask] = arr[mask]
    return out


def mask_to_bw(mask: np.ndarray) -> Image.Image:
    bw = np.zeros(mask.shape, dtype=np.uint8)
    bw[mask] = 255
    return Image.fromarray(bw, mode="L")


def parse_hex_color(hex_str: str) -> Tuple[int, int, int]:
    s = hex_str.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"Cor inválida: {hex_str}")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def make_overlay(
    arr: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float
) -> np.ndarray:
    arr = ensure_rgb(arr)
    img = arr.astype(np.float32)
    out = img.copy()
    color_vec = np.array(color, dtype=np.float32).reshape((1, 1, 3))
    out[mask] = (1.0 - alpha) * img[mask] + alpha * color_vec
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# -------------------- Tesseract helpers --------------------
def detect_tesseract_cmd(user_cmd: Optional[str]) -> Optional[str]:
    if user_cmd:
        return user_cmd
    exe = None
    system = platform.system().lower()
    candidates: List[str] = []
    if system == "darwin":
        candidates += ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]
    elif system == "windows":
        candidates += [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
    else:
        candidates += ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]
    for c in candidates:
        if Path(c).exists():
            exe = c
            break
    return exe


def detect_tessdata_dir(user_dir: Optional[str]) -> Optional[str]:
    if user_dir:
        return user_dir
    system = platform.system().lower()
    candidates: List[str] = []
    if system == "darwin":
        candidates += ["/opt/homebrew/share/tessdata", "/usr/local/share/tessdata"]
    elif system == "windows":
        candidates += [
            r"C:\Program Files\Tesseract-OCR\tessdata",
            r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
        ]
    else:
        candidates += [
            "/usr/share/tesseract-ocr/5/tessdata",
            "/usr/share/tesseract-ocr/4.00/tessdata",
            "/usr/share/tesseract-ocr/tessdata",
        ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


# -------------------- OCR --------------------
def ocr_pages_to_text(
    pages: List[Image.Image],
    lang: str,
    tesseract_cmd: Optional[str],
    tessdata_dir: Optional[str],
) -> str:
    try:
        import pytesseract
    except Exception as e:
        raise RuntimeError(
            "pytesseract não está instalado. Adicione 'pytesseract' ao requirements e instale o Tesseract no SO."
        ) from e

    cmd = detect_tesseract_cmd(tesseract_cmd)
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    td = detect_tessdata_dir(tessdata_dir)
    if td:
        os.environ["TESSDATA_PREFIX"] = td

    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception:
        available = set()

    requested = [x.strip() for x in (lang or "eng").split("+") if x.strip()]
    use_lang = "+".join(requested) if not available else None
    if available:
        keep = [x for x in requested if x in available]
        if not keep:
            keep = ["eng"] if "eng" in available else [next(iter(available))]
        use_lang = "+".join(keep)

    chunks: List[str] = []
    for idx, im in enumerate(pages, start=1):
        txt = pytesseract.image_to_string(im.convert("RGB"), lang=use_lang)
        chunks.append(f"----- Página {idx} -----\n{txt}".rstrip())
    return "\n\n".join(chunks).strip() + "\n"


# -------------------- Worker --------------------
def worker(args):
    (
        in_root,
        out_root,
        in_pdf,
        dpi,
        th,
        apply_judge,
        prev_removed_dir,
        bg_value,
        prev_skip_empty,
        mask_dir,
        overlay_dir,
        overlay_color,
        overlay_alpha,
        overlay_skip,
        extract_text,
        text_dir,
        ocr_lang,
        tesseract_cmd,
        tessdata_dir,
    ) = args
    try:
        out_main = rel_output_path(in_root, out_root, in_pdf)

        pages = convert_from_path(str(in_pdf), dpi=dpi)
        if not pages:
            raise RuntimeError("PDF sem páginas.")

        removed_pages: List[Image.Image] = []
        preview_removed_pages: List[Image.Image] = []
        mask_pages: List[Image.Image] = []
        overlay_pages: List[Image.Image] = []

        for pil_img in pages:
            arr = np.array(pil_img.convert("RGB"))
            mask = build_mask(arr, th, apply_judge)

            removed_pages.append(
                Image.fromarray(handle_remove_to_white(arr, mask), mode="RGB")
            )

            if prev_removed_dir is not None:
                if (not prev_skip_empty) or mask.any():
                    preview_removed_pages.append(
                        Image.fromarray(
                            make_only_removed(arr, mask, bg_value), mode="RGB"
                        )
                    )
            if mask_dir is not None:
                mask_pages.append(mask_to_bw(mask))
            if overlay_dir is not None:
                if (not overlay_skip) or mask.any():
                    overlay_pages.append(
                        Image.fromarray(
                            make_overlay(arr, mask, overlay_color, overlay_alpha),
                            mode="RGB",
                        )
                    )

        out_main.parent.mkdir(parents=True, exist_ok=True)
        f, r = removed_pages[0], removed_pages[1:]
        f.save(str(out_main), "PDF", resolution=dpi, save_all=True, append_images=r)

        if prev_removed_dir is not None:
            out_prev = rel_output_path(in_root, prev_removed_dir, in_pdf)
            out_prev.parent.mkdir(parents=True, exist_ok=True)
            if preview_removed_pages:
                f, r = preview_removed_pages[0], preview_removed_pages[1:]
                f.save(
                    str(out_prev), "PDF", resolution=dpi, save_all=True, append_images=r
                )
            else:
                Image.new("RGB", (1, 1), color=(bg_value,) * 3).save(
                    str(out_prev), "PDF", resolution=dpi
                )

        if mask_dir is not None:
            out_mask = rel_output_path(in_root, mask_dir, in_pdf)
            out_mask.parent.mkdir(parents=True, exist_ok=True)
            if mask_pages:
                f, r = mask_pages[0], mask_pages[1:]
                f.save(
                    str(out_mask), "PDF", resolution=dpi, save_all=True, append_images=r
                )
            else:
                Image.new("L", (1, 1), color=0).save(
                    str(out_mask), "PDF", resolution=dpi
                )

        if overlay_dir is not None:
            out_overlay = rel_output_path(in_root, overlay_dir, in_pdf)
            out_overlay.parent.mkdir(parents=True, exist_ok=True)
            if overlay_pages:
                f, r = overlay_pages[0], overlay_pages[1:]
                f.save(
                    str(out_overlay),
                    "PDF",
                    resolution=dpi,
                    save_all=True,
                    append_images=r,
                )
            else:
                Image.new("RGB", (1, 1), color=(0, 0, 0)).save(
                    str(out_overlay), "PDF", resolution=dpi
                )

        if extract_text:
            out_txt = rel_output_path(in_root, text_dir, in_pdf).with_suffix(".txt")
            out_txt.parent.mkdir(parents=True, exist_ok=True)
            try:
                text_content = ocr_pages_to_text(
                    removed_pages,
                    lang=ocr_lang,
                    tesseract_cmd=tesseract_cmd,
                    tessdata_dir=tessdata_dir,
                )
            except Exception as e:
                text_content = f"[OCR ERRO] {e}\n"
            with open(out_txt, "w", encoding="utf-8") as fh:
                fh.write(text_content)

        return (
            True,
            f"[OK] {in_pdf} -> {out_main} ({len(removed_pages)} pág.)"
            + (f" +TXT" if extract_text else ""),
        )
    except Exception as e:
        return (False, f"[ERRO] {in_pdf}: {e}")


# -------------------- Util --------------------
def discover_pdfs(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.rglob("*.pdf") if p.is_file())


def rel_output_path(input_dir: Path, output_dir: Path, file_path: Path) -> Path:
    rel = file_path.relative_to(input_dir)
    return output_dir / rel.with_suffix(".pdf")


# -------------------- CLI --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove pixels por faixa RGB (pintando de branco) e opcionalmente gera previews e TXT (OCR). Recursivo + paralelo."
    )
    p.add_argument("--input_dir", "-i", type=Path, default=Path("Input"))
    p.add_argument("--output_dir", "-o", type=Path, default=Path("Output"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--apply-judge", action="store_true")

    # Limites RGB (padrão global 170..250)
    p.add_argument("--rgb-min", type=int, default=170)
    p.add_argument("--rgb-max", type=int, default=250)
    p.add_argument("--rmin", type=int)
    p.add_argument("--rmax", type=int)
    p.add_argument("--gmin", type=int)
    p.add_argument("--gmax", type=int)
    p.add_argument("--bmin", type=int)
    p.add_argument("--bmax", type=int)
    p.add_argument("--inclusive", action="store_true")

    # Previews opcionais
    p.add_argument("--preview-removed", action="store_true")
    p.add_argument("--removed-dir", type=Path, default=Path("OutputRemoved"))
    p.add_argument("--bg", choices=["black", "white"], default="black")
    p.add_argument("--skip-empty", action="store_true")

    p.add_argument("--preview-mask", action="store_true")
    p.add_argument("--mask-dir", type=Path, default=Path("OutputMask"))

    p.add_argument("--overlay", action="store_true")
    p.add_argument("--overlay-dir", type=Path, default=Path("OutputOverlay"))
    p.add_argument("--overlay-color", type=str, default="#ff0000")
    p.add_argument("--overlay-alpha", type=float, default=0.6)
    p.add_argument("--overlay-skip-empty", action="store_true")

    # OCR/Tesseract
    p.add_argument(
        "--extract-text",
        action="store_true",
        help="Extrai texto (OCR) dos PDFs processados para OutputTXT.",
    )
    p.add_argument("--text-dir", type=Path, default=Path("OutputTXT"))
    p.add_argument(
        "--ocr-lang",
        type=str,
        default="por",
        help="Idiomas (ex.: 'por', 'eng', 'por+eng').",
    )
    p.add_argument(
        "--tesseract-cmd",
        type=str,
        default="",
        help="Caminho do executável tesseract, se precisar.",
    )
    p.add_argument(
        "--tessdata-dir",
        type=str,
        default="",
        help="Diretório 'tessdata' (define TESSDATA_PREFIX).",
    )

    return p.parse_args()


def build_thresholds_from_args(args: argparse.Namespace) -> RGBThresholds:
    rmin = args.rmin if args.rmin is not None else args.rgb_min
    rmax = args.rmax if args.rmax is not None else args.rgb_max
    gmin = args.gmin if args.gmin is not None else args.rgb_min
    gmax = args.gmax if args.gmax is not None else args.rgb_max
    bmin = args.bmin if args.bmin is not None else args.rgb_min
    bmax = args.bmax if args.bmax is not None else args.rgb_max
    th = RGBThresholds(rmin, rmax, gmin, gmax, bmin, bmax, args.inclusive)
    th.validate()
    return th


def main() -> int:
    args = parse_args()
    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir
    dpi: int = args.dpi
    workers: int = max(1, args.workers)
    apply_judge: bool = args.apply_judge

    bg_value = 0 if args.bg == "black" else 255
    th = build_thresholds_from_args(args)

    try:
        overlay_color = parse_hex_color(args.overlay_color)
    except Exception as e:
        print(f"ERRO cor overlay: {e}", file=sys.stderr)
        return 2
    overlay_alpha = float(args.overlay_alpha)
    if not (0.0 <= overlay_alpha <= 1.0):
        print("ERRO: --overlay-alpha deve estar em [0,1].", file=sys.stderr)
        return 2

    if not in_dir.exists():
        print(f"ERRO: pasta de entrada não existe: {in_dir}", file=sys.stderr)
        return 2

    pdfs = discover_pdfs(in_dir)
    if not pdfs:
        print(f"Nenhum PDF encontrado em {in_dir}.")
        return 0

    print(
        f"{len(pdfs)} PDF(s) | Limites R[{th.rmin},{th.rmax}] G[{th.gmin},{th.gmax}] B[{th.bmin},{th.bmax}] "
        f"{'inclusive' if th.inclusive else 'exclusivo'} | workers={workers} | OCR={'on' if args.extract_text else 'off'}"
    )

    prev_removed_dir = args.removed_dir if args.preview_removed else None
    mask_dir = args.mask_dir if args.preview_mask else None
    overlay_dir = args.overlay_dir if args.overlay else None
    tesseract_cmd = args.tesseract_cmd or None
    tessdata_dir = args.tessdata_dir or None

    tasks = []
    for p in pdfs:
        tasks.append(
            (
                in_dir,
                out_dir,
                p,
                dpi,
                th,
                apply_judge,
                prev_removed_dir,
                bg_value,
                args.skip_empty,
                mask_dir,
                overlay_dir,
                overlay_color,
                overlay_alpha,
                args.overlay_skip_empty,
                args.extract_text,
                args.text_dir,
                args.ocr_lang,
                tesseract_cmd,
                tessdata_dir,
            )
        )

    ok = fail = 0
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for i, fut in enumerate(cf.as_completed(futures), start=1):
            success, msg = fut.result()
            print(f"[{i}/{len(futures)}] {msg}")
            ok += int(success)
            fail += int(not success)

    print(
        f"Concluído. Sucesso: {ok} | Falhas: {fail} | Output: {out_dir.resolve()}"
        + (f" | TXT: {args.text_dir.resolve()}" if args.extract_text else "")
        + (f" | Removed: {args.removed_dir.resolve()}" if args.preview_removed else "")
        + (f" | Mask: {args.mask_dir.resolve()}" if args.preview_mask else "")
        + (f" | Overlay: {args.overlay_dir.resolve()}" if args.overlay else "")
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
