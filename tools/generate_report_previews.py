#!/usr/bin/env python3
"""Generate lightweight report preview HTML and PDF artifacts for docs."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "report-previews"


EN_LINES = [
    "AnyChain Benchmark Agent - Report Preview",
    "This preview shows the report sections users will see after a benchmark run.",
    "Executive Summary",
    "The report starts with run metadata, cloud provider, region, machine type, disk baseline, RPC mode, and maximum stable QPS.",
    "Core Performance Charts",
    "Performance overview explains CPU, memory, disk IOPS, throughput, and utilization trends.",
    "CPU-disk correlation helps identify whether high CPU time is actually disk wait or queue depth.",
    "Disk threshold charts compare observed utilization and latency with configured IOPS and throughput baselines.",
    "Per-Method RPC Attribution",
    "Per-method charts show success/failure counts and P50/P90/P99 latency for workload RPC methods only.",
    "Sync Health",
    "Sync-health charts show height gap, reported lag, or freshness depending on the chain family.",
    "Artifact Evidence",
    "The Agent cites HTML, CSV, runtime.env, archive summary, and optional Prometheus/Grafana endpoints.",
]

ZH_LINES = [
    "AnyChain Benchmark Agent - 报告预览",
    "该预览展示 benchmark 运行后用户会看到的报告结构。",
    "执行摘要",
    "报告首先展示运行元数据、云厂商、区域、机器类型、磁盘基线、RPC 模式和最大稳定 QPS。",
    "核心性能图表",
    "Performance overview 解释 CPU、内存、磁盘 IOPS、吞吐和利用率趋势。",
    "CPU-disk correlation 用于判断 CPU 压力是否来自磁盘等待或队列深度。",
    "Disk threshold 图表会将实际利用率和延迟与配置的 IOPS、吞吐基线对比。",
    "Per-Method RPC 归因",
    "Per-method 图表只统计 workload RPC method 的成功/失败次数和 P50/P90/P99 延迟。",
    "同步健康",
    "同步健康图表根据不同链 family 展示高度差、reported lag 或 freshness。",
    "证据文件",
    "Agent 会引用 HTML、CSV、runtime.env、归档 summary，以及可选 Prometheus/Grafana endpoint。",
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _write_html(OUT / "report-preview.en.html", "AnyChain Report Preview", EN_LINES, "en")
    _write_html(OUT / "report-preview.zh.html", "AnyChain 报告预览", ZH_LINES, "zh")
    _write_pdf(OUT / "report-preview.en.pdf", EN_LINES, cjk=False)
    _write_pdf(OUT / "report-preview.zh.pdf", ZH_LINES, cjk=True)
    return 0


def _write_html(path: Path, title: str, lines: list[str], lang: str) -> None:
    cards = "\n".join(
        f"<section><h2>{lines[idx]}</h2><p>{lines[idx + 1]}</p><div class='bars'><span style='width:{55 + idx * 3 % 35}%'></span></div></section>"
        for idx in range(2, len(lines) - 1, 2)
    )
    path.write_text(f"""<!doctype html>
<html lang="{lang}">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; margin: 40px; color: #182230; background: #f6f8fb; }}
    main {{ max-width: 980px; margin: auto; }}
    h1 {{ font-size: 38px; margin-bottom: 8px; }}
    .subtitle {{ color: #526173; font-size: 18px; margin-bottom: 28px; }}
    section {{ background: white; border: 1px solid #d9e1ec; border-radius: 8px; padding: 20px; margin: 16px 0; }}
    h2 {{ margin: 0 0 10px; font-size: 22px; }}
    p {{ line-height: 1.55; }}
    .bars {{ height: 12px; background: #e8eef6; border-radius: 999px; overflow: hidden; }}
    .bars span {{ display: block; height: 100%; background: linear-gradient(90deg, #2563eb, #16a34a); }}
  </style>
</head>
<body><main>
  <h1>{lines[0]}</h1>
  <p class="subtitle">{lines[1]}</p>
  {cards}
</main></body></html>
""", encoding="utf-8")


def _write_pdf(path: Path, lines: list[str], cjk: bool = False) -> None:
    objects: list[bytes] = []
    pages = []
    page_chunks = [lines[idx:idx + 7] for idx in range(0, len(lines), 7)]
    font_obj_num = 3
    for chunk in page_chunks:
        content = _page_content(chunk, cjk)
        content_obj = len(objects) + 4
        page_obj = len(objects) + 5
        objects.append(f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"\nendstream")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_obj_num} 0 R >> >> /Contents {content_obj} 0 R >>".encode("ascii")
        )
        pages.append(page_obj)
    if cjk:
        font = b"<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H /DescendantFonts [<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light /CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 2 >> >>] >>"
    else:
        font = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    kids = " ".join(f"{page} 0 R" for page in pages)
    base_objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"),
        font,
    ]
    all_objects = base_objects + objects
    _write_pdf_objects(path, all_objects)


def _page_content(lines: list[str], cjk: bool) -> bytes:
    parts = [b"BT", b"/F1 18 Tf", b"72 730 Td"]
    for idx, line in enumerate(lines):
        if idx == 1:
            parts.append(b"/F1 11 Tf")
        elif idx > 1 and idx % 2 == 0:
            parts.append(b"/F1 14 Tf")
        else:
            parts.append(b"/F1 11 Tf")
        if cjk:
            encoded = line.encode("utf-16-be").hex().upper()
            parts.append(f"<{encoded}> Tj".encode("ascii"))
        else:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            parts.append(f"({escaped}) Tj".encode("latin-1", errors="replace"))
        parts.append(b"0 -38 Td")
    parts.append(b"ET")
    return b"\n".join(parts)


def _write_pdf_objects(path: Path, objects: list[bytes]) -> None:
    content = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{idx} 0 obj\n".encode("ascii"))
        content.extend(obj)
        content.extend(b"\nendobj\n")
    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(bytes(content))


if __name__ == "__main__":
    raise SystemExit(main())
