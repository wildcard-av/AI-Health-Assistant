"""Human-readable package version (bump manually for releases if you want).

On each push to ``main`` / ``master``, GitHub Actions also creates an automatic
**git tag** (e.g. ``v0.1.42``) that increments the patch level. Prefer tags for
exact CI/build identity when debugging."""

__version__ = "0.1.0"
