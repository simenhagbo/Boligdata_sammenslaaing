"""
Felles tilgang til hemmeligheter (API-nøkler) for collect-modulene.

Leter etter verdier i to steder, i prioritet:
  1. Miljøvariabler (typisk satt i shell eller CI)
  2. .env-fil i prosjektrot (manuelt redigert, ignoreres av git)

.env-formatet er enkelt: én linje per nøkkel:
    FROST_CLIENT_ID=din-id-her
    # kommentarer ignoreres

Brukes for kilder som krever auth (Frost). Åpne kilder (SSB, Norges Bank,
Entur, Kartverket) trenger ingenting av dette.
"""

import os
from pathlib import Path


def _read_env_file(env_path: Path) -> dict[str, str]:
    """Parse en enkel KEY=value-fil. Ignorerer tomme linjer og kommentarer."""
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        # Tillat omkringliggende anførselstegn slik at FOO="bar" og FOO=bar
        # begge gir verdien "bar"
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def get_secret(key: str) -> str | None:
    """Hent en hemmelighet ved navn. Returnerer None hvis ikke satt noe sted.

    Miljøvariabel har høyest prioritet — slik at CI-systemer og automatiserte
    kjøringer kan injisere verdier uten å skrive en .env-fil.
    """
    # Sjekk miljøvariabler først
    val = os.environ.get(key)
    if val:
        return val.strip()

    # Fall tilbake til .env-fil i prosjektrot
    env_path = Path(__file__).parents[2] / ".env"
    env_values = _read_env_file(env_path)
    return env_values.get(key)
