"""ReviewLens: semi-supervised sentiment classification of Persian reviews, with an AI agent."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("reviewlens")
except PackageNotFoundError:  # running from a source tree that was not installed
    __version__ = "0.0.0"
