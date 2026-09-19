---
name: python-code-writer
description: Python style rules for this repo. MUST be used before writing, editing or refactoring any Python code (.py files, notebooks, snippets) — covers simplicity, type hints, NumPy-style docstrings with Parameters and Returns, and cohesive classes.
---

# Python code writer

Follow these rules whenever you write or modify Python code in this repository, whether you are the main agent or a subagent.
Apply them to every function, method and class you add or change.

## Simplicity above all

- Write the simplest code that works. Short functions, obvious control flow, no cleverness.
- Use what the installed libraries already provide (pandas, numpy, rdkit, scikit-learn) before writing your own.
- No defensive branching for cases that haven't come up, no speculative parameters, no unused abstractions.
- If a function grows long or does two things, split it.

## Type hints

Always add type hints to parameters and return values where possible, including `-> None`.
Use built-in generics (`list[str]`, `dict[str, int]`) and `X | None` rather than `typing.List` / `Optional`.

## Docstrings

Every function and method gets a docstring with:

1. **A brief, clear overview** — one or two sentences so the reader understands completely what the function does (and why, if that's not obvious).
2. **Parameters** — each argument, its type, and what it means.
3. **Returns** — the type and what it represents.

Use NumPy style:

```python
def canonicalise(smiles: str) -> str | None:
    """
    Convert a SMILES string to RDKit's canonical form so duplicate molecules
    written differently compare equal.

    Parameters
    ----------
    smiles : str
        SMILES string to canonicalise.

    Returns
    -------
    str | None
        Canonical SMILES, or None if RDKit cannot parse the input.
    """
```

Omit `Parameters` when there are none (besides `self`) and `Returns` when the function returns `None`.

The docstring must be a standalone description of the function, not a record of the model's reasoning while writing it. Don't reference the current task, prior versions, alternatives considered, or planning process ("switched to this approach because...", "fixes the bug from..."); a new reader with no context on how the code came to exist must be able to read the docstring and fully understand the function.

## Classes

- The class docstring states the **purpose** of the class and documents its constructor arguments in a `Parameters` section.
- Every method must relate directly to that purpose. If a method doesn't fit, it belongs in another class or as a module-level function.

```python
class Eda:
    """
    Produce exploratory figures for one target's preprocessed training data.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed frame with `smiles` and `labels` columns.
    target : str
        Short target name, used in figure filenames.
    """

    def __init__(self, df: pd.DataFrame, target: str) -> None:
        self.df = df
        self.target = target
```
