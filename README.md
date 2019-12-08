# sips 0.14.2

setup:

- install
- navigate to dir
- local pip install

```bash
git clone https://github.com/anandijain/sips.git
cd sips
pip install -e . [--user]
```

## USAGE

1. get data

    ```python
    from sips.lines import lines
    lines.Lines()
    # or
    from sips.lines.bov import bov
    bov.lines(['nba'])
    ```

2. train LSTM predictor on data

    - place `lines.py` output CSVs in `sips/data/lines`
    - go to `sips/ml/` and `python ml_pred.py`

## CHANGELOG / ROADMAP

1. the sports-reference api has been largely revamped

2. preparing for pypi release

3. post will be moved to separate repo and have sips as dependency

4. premature update of master to 0.14.2

5. 0.14.2 fixes some relative import problems, could be lingering instability
