# TraceMap

TraceMap is a small vision library for tracing a prediction back to the
training images that nudged it in one direction or another.

You give it a query image. It gives you the most helpful and harmful training
examples, a heatmap on the query, a heatmap on each retrieved example, and a
few patch-level matches so the comparison is easier to read.

The scope is deliberately tight. The backbone stays frozen, the classifier head
is trainable, and the attribution math lives at the head. That keeps the code
short enough to understand without turning it into a giant framework.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you want the app:

```bash
pip install -e ".[dev,demo]"
```

## Quick start

```python
from tracemap import TraceMap, TraceMapConfig
from tracemap.data import build_default_pet_datasets

config = TraceMapConfig()
datasets = build_default_pet_datasets(config, download=True)

tm = TraceMap(config)
tm.fit(datasets.train, datasets.val)
tm.build_index(datasets.train)

image, _ = datasets.test[0]
result = tm.explain(image, top_k=3)

print(result.prediction.class_name, result.prediction.confidence)
print(result.helpful_examples[0].influence_score)
```

By default the example setup uses five Oxford-IIIT Pet classes:
`Abyssinian`, `Birman`, `Persian`, `Beagle`, and `Chihuahua`.

## App

The Streamlit app is the easiest way to poke around:

```bash
streamlit run app/streamlit_app.py
```

## Development

There is a small but normal test and lint setup:

```bash
ruff check src tests app
pytest
python -m compileall src tests app
```

If you like Make targets:

```bash
make install
make lint
make test
make app
```

## Notes

This is not trying to be a full attribution toolkit. It is one focused idea,
built to be readable and easy to extend.

- The backbone is frozen.
- Influence is measured over the linear head.
- The spatial output is best read as evidence maps plus patch matches, not exact
  pixel-to-pixel correspondence.
