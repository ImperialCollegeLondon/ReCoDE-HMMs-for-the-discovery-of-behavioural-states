# Development

Set up your environment then install dependencies from `dev-requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate # with Powershell on Windows: `.venv\Scripts\Activate.ps1`
```

```bash
pip install -r dev-requirements.txt
```

Install the git hooks:

```bash
pre-commit install
```

## Dependencies

Dependencies are managed using the [`pip-tools`] tool chain.

Unpinned dependencies are specified in `pyproject.toml`. Pinned versions are
then produced with:

```sh
pip-compile pyproject.toml
```

To add/remove packages edit `pyproject.toml` and run the above command. To
upgrade all existing dependencies run:

```sh
pip-compile --upgrade pyproject.toml
```

Dependencies for developers are listed separately as optional, with the pinned versions
being saved to `dev-requirements.txt` instead.

`pip-tools` can also manage these dependencies by adding extra arguments, e.g.:

```sh
pip-compile -o dev-requirements.txt --extra=dev pyproject.toml
```

When dependencies are upgraded, both `requirements.txt` and `dev-requirements.txt`
should be regenerated so that they are in sync.

[`pip-tools`]: https://github.com/jazzband/pip-tools
