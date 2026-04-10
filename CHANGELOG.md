# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed

- Improved wrapped-model compatibility for tracing and analysis utilities.
- Added support for tuple-style layer outputs by consistently operating on the primary tensor while preserving auxiliary values.
- Improved module and layer resolution for nested wrapper layouts such as `language_model.model.*`, including non-default layer attributes like `h`.
- Added regression coverage for wrapped Llama-, Qwen-, and Gemma-style compatibility paths.
