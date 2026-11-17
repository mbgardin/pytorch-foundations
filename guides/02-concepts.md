# Concepts and common mistakes

## From sklearn to PyTorch

An sklearn `Pipeline` applies the same fitted transformations at prediction time and an estimator typically owns `fit` and `predict`. A PyTorch `nn.Module` defines a forward computation; it does not split data, fit a scaler, choose a loss, or train itself. A `Dataset` maps an index to one example, while a `DataLoader` groups examples into batches. An optimizer owns the update rule and state such as momentum. You write the loop tying them together.

Split before fitting preprocessing. Estimate means, standard deviations, category mappings, and other learned transformations from training data only; reuse those values for validation, test, and new observations. Tune choices with validation data, and use the test set only after choices are finalized. Grouped or time-ordered data may require a group or time-based split instead of a random split.

## Shape and objective

For a batch of `B` rows and `D` features, inputs usually have shape `(B, D)`. For `C` classes, a classifier can return logits of shape `(B, C)` and integer labels of shape `(B,)` with dtype `torch.long`. `CrossEntropyLoss` expects raw logits, not softmax probabilities. Use `argmax(dim=1)` for class predictions and `softmax(dim=1)` when probabilities are needed. Binary classification can instead use one logit with `BCEWithLogitsLoss`, with floating targets of matching shape. Regression commonly uses floating targets and a loss such as `MSELoss`; align output and target shapes to avoid unintended broadcasting.

## One training step

1. Set `model.train()` for training; this activates dropout and changes batch normalization behavior.
2. Get a batch; move both inputs and targets to the same device as the model.
3. `optimizer.zero_grad()` clears accumulated parameter gradients.
4. Compute predictions and a scalar loss; call `loss.backward()` to compute gradients.
5. `optimizer.step()` updates parameters. An epoch is one pass over the training loader, not one update; the final batch may be smaller.

For validation and inference, call `model.eval()` and use `torch.no_grad()` (or inference mode). These do different jobs: evaluation mode changes the behavior of certain layers; disabling gradients avoids building a gradient graph. Average a mean batch loss weighted by the batch size to get the mean over examples, and aggregate correct predictions across all examples. Restore `model.train()` before training again.

## Reproducibility and diagnosis

Seeds help reproduce runs but do not guarantee identical results across platforms, devices, versions, or every operation. Check tensor shapes, dtypes, target ranges, missing values, finite losses, learning rate, and whether the correct split and model mode are used. First try to overfit a tiny training subset as a debugging exercise. Compare against a simple baseline, use suitable metrics for class imbalance, and look at learning curves. A model can memorize synthetic patterns without demonstrating real-world performance.

## More to explore

After the basics: convolutional and sequence models, embeddings, learning-rate schedulers, class weighting, transfer learning, cross-validation, more careful reproducibility, custom `collate_fn`, and acceleration. These are extensions, not prerequisites for the included notebooks.
