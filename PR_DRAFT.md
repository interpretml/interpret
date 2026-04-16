Fixes #635 Repeated Callback Iterations

## Description
This PR resolves an issue during EBM training where the `callback` parameter is invoked repeatedly with the exact same `n_steps` value whenever the boosting loop fails to make progress.

### The Bug
In the boosting loop inside `_boost.py`, the callback was being unconditionally invoked at the very end of the loop, regardless of whether `make_progress` was `True` or `False`. Because `step_idx` is only incremented when `make_progress` is `True`, non-progressing cycle iterations continuously spammed the callback with the exact same `n_steps` value, resulting in repeating redundant output logs for the user.

### The Fix
I moved the callback invocation **inside** the `if make_progress:` block, right after early stopping tolerance evaluations.
```diff
-                        if callback is not None:
-                            is_done = callback(
-                                bag_idx, step_idx, make_progress, cur_metric
-                            )
-                            if is_done:
-                                if stop_flag is not None:
-                                    stop_flag[0] = True
-                                break
```
```python
+                            if callback is not None:
+                                is_done = callback(
+                                    bag_idx, step_idx, make_progress, cur_metric
+                                )
+                                if is_done:
+                                    if stop_flag is not None:
+                                        stop_flag[0] = True
+                                    break
```

This change strictly guarantees that the callback only receives monotonic, progressing `n_steps` values.

## Design Decisions
* The `has_progressed` parameter has been strictly maintained for API backward compatibility. Because the callback is now only ever fired after progress is made, it will always be evaluated as `True`. This ensures any pre-existing user callbacks utilizing `if has_progressed:` do not unexpectedly break.

## Testing
Added 5 robust regression tests inside a new `test_callback.py` file to maintain the invariant:
1. `test_callback_no_repeated_steps_classifier`: Verifies `n_steps` remains monotonically increasing across distinct boosting phases (main terms vs interaction pairs) without any duplicate step integers.
2. `test_callback_no_repeated_steps_regressor`: Equivalent functionality verification for the regressor implementation.
3. `test_callback_has_progressed_always_true`: Validates `has_progressed` correctly always outputs `True`.
4. `test_callback_early_termination`: Assertions establishing that `is_done = True` continues to properly bypass early stopping settings to instantly conclude training.
5. `test_callback_receives_valid_metrics`: Ensures the `best_score` metric received by the callback does not evaluate to NaN or infinite sequences.

All existing and newly included formatting validation and regression tests successfully pass without issue.
