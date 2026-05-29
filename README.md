# sinclair

`sinclair` is a small opinionated runtime for analytical products.

It is built for teams that need answers that can be published, inspected, and reused in product, not just generated once in a notebook.

## Shape

- `sinclair`: the core runtime for structured agents
- `sinclair.survey`: the survey module for reports, chat, evidence, and frontend hydration

## Why it exists

Most agent stacks are good at producing text.

They are much worse at producing work that survives contact with a real product: stable outputs, repeatable evidence, frontend-safe references, and a clear path from raw data to something a user can click.

`sinclair` is for that layer.

## What it is good at

- structured runs with explicit completion
- controlled tool use
- dataframe-backed analysis
- survey reports that stay tied to evidence
- chat over the same analytical state
- frontend bundles with stable references and provenance

## Install

```bash
uv add sinclair
```

Or straight from Git:

```bash
uv add git+ssh://git@github.com/jovemexausto/sinclair.git
```

## Example

```python
from sinclair.survey import SurveyDefaults, SurveyIdentityPolicy, SurveyStudy


study = SurveyStudy.from_csv(
    "study.csv",
    study_id="nps-q2",
    defaults=SurveyDefaults(
        model="gpt-5.4",
        identity=SurveyIdentityPolicy(respondent_id_column="respondent_id"),
    ),
)

report = study.report_question("Q10")
bundle = study.frontend_controller().render_report("question:Q10")
```

## Product idea

The model should not carry product complexity alone.

The runtime should help turn model output into something operational:

1. a report is produced in a constrained shape
2. evidence stays attached to claims
3. references become stable product objects
4. the frontend receives material it can render and inspect safely

## Development

```bash
PYTHONPATH=. uv run --with pytest python -m pytest
```

## License

MIT
