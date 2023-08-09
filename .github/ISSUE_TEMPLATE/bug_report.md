---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is. If you are referring to methods reported in papers please add a link to the paper (preferably Arxiv).

**To Reproduce**
Steps to reproduce the behavior: Ideally an executable code snipped like
```python
import tequila as tq

U = tq.gates.Ry(angle="a", target=0)
H = tq.paulis.X(0)
E = tq.ExpectationValue(H=H, U=U)

energy = tq.compile(E)

eval = energy()

```

**Expected behavior**
A clear and concise description of what you expected to happen or what you want to do.

**Computer (please complete the following information):**
 - OS: [e.g. iOS/Linux/Windows]
 - Version [Python and Tequila]
Get is via
```
import sys, platform
import tequila as tq

print("tequila version: ", tq.__version__)
print("python version: ", sys.version)
print("platform: ", platform.version())  

```

**Smartphone (please complete the following information):**
 - Device: [e.g. iPhone6]
 - OS: [e.g. iOS8.1]
 - Browser [e.g. stock browser, safari]
 - Version [e.g. 22]

**Additional context**
Add any other context about the problem here.
