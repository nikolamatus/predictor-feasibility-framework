# Extending the framework

Plugins are registered in process with a decorator. This version does not discover plugins through package entry points. Import the module that calls the decorator before `run`, typically from the study module.

## Reader

Built-ins are `csv` and `parquet`. A reader takes a path and returns a data frame.

```python
from pathlib import Path

import pandas as pd

from feasibility_framework.registries import register_reader


@register_reader("fixed")
def read_fixed(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="|")
```

`load_data` is still your hook. Call `get_reader("fixed")(path)` from that hook. The engine then checks the identifier, joins sources, and applies the leakage guard. Registering a second function under an existing name raises.

### Non-file data sources

`load_data(self, path_map)` only has to return `dict[str, pandas.DataFrame]`. Nothing requires those frames to come from a path. `path_map` exists for the common case of file-backed sources; a study is free to ignore it and build its frames another way, for example:

```python
def load_data(self, path_map: dict[str, Path]) -> dict[str, pd.DataFrame]:
    import pandas as pd
    import snowflake.connector

    with snowflake.connector.connect(...) as conn:
        primary = pd.read_sql("select ... from ...", conn)
    return {"primary": primary}
```

The reader registry above is a convenience for files, not a constraint on where data comes from. Everything downstream, the leakage guard, linkage, EPV gate, still runs the same way regardless of how the frame was produced.

## Composite strategy

Built-ins are `sum-score` and `mean-score`. A strategy receives the frame and the block's source column names, and returns one series. It runs only when a block sets `max_categorical_levels` and an observed categorical column exceeds that cap. If the cap is exceeded and `composite_strategy` is empty, feature construction raises. The strategy name is part of the locked config, so choosing one after an underpowered run means locking again.

```python
import pandas as pd

from feasibility_framework.registries import register_composite


@register_composite("count-of-endorsed")
def count_of_endorsed(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return (numeric > 0).sum(axis=1)
```

Point the block at it with `composite_strategy="count-of-endorsed"`. The engine leakage-checks the source columns, not the score name alone.

## Inference method

The built-in name is `nadeau_bengio_holm`. A method is a class with `compare(self, folds, contrasts)`. `folds` is one row per cross-validation split and includes `n_cv_train` and `n_cv_val`. Each contrast names a column of paired differences. `run` calls `self.inference_method.compare(...)` and does not mention a particular test.

```python
from feasibility_framework.registries import InferenceResult, register_inference


@register_inference("my-method")
class MyMethod:
    name = "my-method"

    def compare(self, folds, contrasts):
        rows = []
        for spec in contrasts:
            rows.append(
                {
                    "contrast_id": spec.contrast_id,
                    "metric": spec.metric,
                    "mean_delta": float(folds[spec.column].mean()),
                }
            )
        return InferenceResult(method=self.name, rows=rows)
```

Set `inference_method="my-method"` on the study config before locking.
