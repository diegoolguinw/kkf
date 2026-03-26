---
name: Bug Report
about: Report a bug to help us improve KKF
title: "[BUG] "
labels: bug
assignees: ""
---

## Description

A clear and concise description of what the bug is.

## Steps to Reproduce

Steps to reproduce the behavior:
1. Setup code / system...
2. Execute...
3. See error...

## Minimal Reproducible Example

```python
import numpy as np
from scipy import stats
from KKF import DynamicalSystem, KoopmanOperator
from KKF.applyKKF import apply_koopman_kalman_filter

# Your minimal code that reproduces the issue
```

## Expected Behavior

What should have happened?

## Actual Behavior

What actually happened? Include any error messages and full traceback:

```
Traceback (most recent call last):
  ...
```

## Environment

- **Python version**: [e.g. 3.10]
- **KKF version**: [e.g. 0.2.0]
- **Operating System**: [e.g. Ubuntu 22.04, Windows 11, macOS 13]
- **NumPy version**: [e.g. 1.23.0]
- **SciPy version**: [e.g. 1.9.0]
- **scikit-learn version**: [e.g. 1.1.0]

## Additional Context

Any other context about the problem here.

## Checklist

- [ ] I have checked that this issue has not already been reported
- [ ] I have provided a minimal reproducible example
- [ ] I have included the full error traceback
- [ ] I have specified my environment details
