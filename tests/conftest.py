"""Shared test configuration."""

import matplotlib

# Force a headless backend: analysis helpers call plt.show(), which blocks
# the suite on a native window with the default macOS backend.
matplotlib.use("Agg")
