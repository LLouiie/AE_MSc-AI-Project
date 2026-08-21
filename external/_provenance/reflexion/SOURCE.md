# external/reflexion — provenance

Upstream: https://github.com/noahshinn/reflexion.git
Pinned commit: `218cf0ef1df84b05ce379dd4a8e47f17766733a0` (2025-01-13)

Cloned as an **independent version anchor** — a reference for what the official
implementation does, so our own `ae/baselines/reflexion.py` can be checked against it.
The Reflexion arm in our experiments runs through our runner
(`--baseline reflexion --max-trials 4 --demo-config configs/demos/one_shot_v1.yaml`),
not through this clone.

```bash
git clone https://github.com/noahshinn/reflexion.git external/reflexion
cd external/reflexion
git checkout 218cf0ef1df84b05ce379dd4a8e47f17766733a0
```

No local modifications.
