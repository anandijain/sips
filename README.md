# sips 0.14

setup:

- install
- navigate to dir
- local pip install

```bash
git clone https://github.com/anandijain/sips.git
cd sips
pip install -e . [--user]
```

usages:

1. gather data

    ```python
    from sips.lines import lines
    lines.Lines()
    ```

2. train on data

    - place lines.py CSVs in sips/data/lines
    - go to sips/ml/ and `python tf_lstm.py`
