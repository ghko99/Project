# Template Usage Notes

Use this repository as a small classification workflow template. Before adapting it to a new dataset, make the data and metric assumptions explicit.

## Setup Checklist

- Install dependencies from `requirements.txt`.
- Update the input path in `template.py`.
- Confirm the target column and feature columns.
- Check missing values before train/test splitting.
- Save the random seed used for the split.

## Reporting

For each run, record the classifier, preprocessing choices, train/test split size, and reported metrics. Keep generated CSV outputs or model artifacts outside Git unless they are intentionally small examples.
