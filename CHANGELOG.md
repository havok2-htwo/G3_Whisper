# Changelog

Dieses Dokument wird bewusst knapp gehalten und bei kuenftigen Arbeiten fortgeschrieben.

## Unreleased

- Modell-Loading fuer lokalen ASR-Pfad so umgestellt, dass der Single-GPU-/CPU-Betrieb nicht mehr indirekt `accelerate` ueber `device_map` erzwingt.
- Benchmark und API geben bei Ladefehlern jetzt die konkrete Loader-Ursache weiter.
- Speech-only-Vorfilterung fuer den optionalen Stimmvektor dokumentiert und aktiv im Runtime-Pfad verankert.
- Benchmark-/Batch-Pfad mit `torch.compile(...)` repariert, indem Modell- und Processor-Checks auf explizite `is None`-Pruefungen umgestellt wurden.
- Benchmark- und Metrikwerte im Admin-Frontend auf deutsches Zahlenformat und besseres Umbruchverhalten umgestellt.
- `librosa` als explizite Python-Abhaengigkeit aufgenommen, damit lokale ASR-/Benchmark-Pfade im `venv`-Workflow sauber starten.
- Cohere-ASR-Loader auf `attn_implementation=\"eager\"` umgestellt und Dynamic-Module-Cache fuer `trust_remote_code` in den Projektordner verlegt.
- Cohere-`transcribe()` auf Windows so abgesichert, dass ohne `cl.exe` der optionale Compile-Pfad automatisch deaktiviert wird.
