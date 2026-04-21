# HTML Spotlight Presentation

Main file:

- [spotlight.html](spotlight.html)

Open it directly in a browser:

```bash
open presentation/spotlight.html
```

Or serve the repo locally:

```bash
python3 -m http.server 8000
```

Then visit:

```text
http://localhost:8000/presentation/spotlight.html
```

The deck is self-contained and uses keyboard navigation:

- `Right Arrow`, `PageDown`, `J`, or `Space`: next slide
- `Left Arrow`, `PageUp`, or `K`: previous slide
- `Home`: first slide
- `End`: last slide

What this deck now covers:

- full project framing, not just the old clarification benchmark
- frozen datasets and split policy
- legacy baseline context and why it is legacy-only
- the new `resample_clarify` method and calibrator
- Phase 1 teacher evaluation, Phase 2 student distillation, and optional Phase 3 DPO
- end-to-end runtime, sharding, docs, and operator workflow
- manuscript status and exact placeholder mapping for unfinished artifacts

Placeholder policy:

- verified artifacts are shown as verified results
- unfinished artifacts are shown as explicit placeholders
- legacy clarification-only outputs are labeled as reference-only, not final pivot results
- the presentation is meant to stay safe to show even while live runs are still in progress
