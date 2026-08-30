---
name: Python runtime paths
description: Package scripts for Python-backed artifact services run from the package directory while the shared Python runtime lives at the workspace root.
---

Use workspace-relative paths from the artifact package to reach the shared Python runtime; do not assume a package-local `.pythonlibs` directory exists.

**Why:** Managed artifact workflows execute package scripts with the package directory as the working directory, while this workspace provisions Python dependencies once at the repository root.

**How to apply:** When editing Python service `dev`, `build`, `start`, or utility scripts, resolve the shared runtime from the package directory before restarting the managed workflow.