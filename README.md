# sips 0.15

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
    - go to `sips/ml/tf_models` and `python ml_pred.py`
    - this might have been broken from 0.14.2 to 0.15

## CHANGELOG / ROADMAP

1. the sports-reference api has been largely revamped

2. pytorch has remerged as the tool of choice
