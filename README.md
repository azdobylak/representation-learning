## Model training
Model is defined in representation.ipynb notebook. Hyperparameters are defined at the top of the notebook.

## Implemented methods

- *triplet_loss_batch_all* - uses all valid triplets in given batch
- *triplet_loss_batch_hard_negative* - for every anchor-positive pair only the hardest negative is used (the closest one)
- *triplet_loss_batch_hard* - for every anchor uses only hardest postive and hardest negative

## Dependencies
```python
tensorflow==1.13
```

## Changelog

#### 01-04-2019
- added comments to notebook
- removed redundant code
