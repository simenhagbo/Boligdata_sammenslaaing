"""
Henter boligprisstatistikk-Excelen fra Eiendom Norge.

URL-en endrer seg månedlig og Cloudflare blokkerer ofte direkte nedlasting.
Hvis det går galt, last ned filen manuelt fra
https://eiendomnorge.no/boligprisstatistikk/ og legg den i
data/raw_data/eiendom_norge/prisstatistikk.xlsx. Pipelinen fortsetter uten
denne kilden hvis filen mangler — du får bare ikke median_pris_m2.
"""

from pathlib import Path

import requests

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "eiendom_norge"
RAW_DIR.mkdir(parents=True, exist_ok=True)

STATISTIKK_URL = (
    "https://eiendomnorge.no/wp-content/uploads/statistikk/boligprisstatistikk.xlsx"
)


def fetch_prisstatistikk() -> None:
    out_path = RAW_DIR / "prisstatistikk.xlsx"
    if out_path.exists():
        print(f"  Allerede hentet: {out_path.name}")
        return

    print("  Henter prisstatistikk fra Eiendom Norge...")
    try:
        resp = requests.get(
            STATISTIKK_URL,
            # Cloudflare blokkerer requests uten User-Agent
            headers={"User-Agent": "Mozilla/5.0 (research/data-pipeline)"},
            timeout=60,
        )
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        print(f"  Lagret: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")
    except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
        print(f"  ADVARSEL: Kunne ikke hente fra Eiendom Norge: {e}")
        print(
            "  Last ned Excel-filen manuelt fra https://eiendomnorge.no/boligprisstatistikk/"
            f"\n  og lagre den til: {out_path}"
            "\n  Pipelinen fortsetter — median_pris_m2 vil mangle i output."
        )


def main() -> None:
    print("=== Innsamling: Eiendom Norge ===")
    fetch_prisstatistikk()
    print("=== Eiendom Norge ferdig ===\n")


if __name__ == "__main__":
    main()
