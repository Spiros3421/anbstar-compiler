# anbstar-compiler
Python implementation of an AnB* compiler that translates structured, stateful protocol specifications into OFMC-compatible AnB models with bounded unfolding and indexed message tagging.

This project implements a structured extension of the AnB language designed to model evolving-key security protocols such as ratchet-based constructions. The compiler translates AnB* specifications into standard AnB models executable by the Open-Source Fixedpoint Model Checker (OFMC).

---

## Overview

AnB* introduces:

- Explicit protocol state
- Sequential local bindings (`Let`)
- Parallel state updates (`New State`)
- Bounded repetition (`Repeat K`)
- Per-iteration message tagging (e.g., `format → format1, format2, ...`)

The compiler implements a compositional translation based on a state environment (σ) and a local environment (ρ), as described in the associated Master’s thesis.

---

## Project Structure

- `anbstar_parser.py` — Tokeniser and recursive-descent parser (AnB* → AST)
- `compile.py` — σ/ρ translation engine and Repeat unfolding
- `emit.py` — Pretty-printer to OFMC-compatible AnB
- `translator.py` — Command-line entry point

---

## Requirements

- Python 3.10 or higher
- No external dependencies

---

## Usage

```bash
python translator.py input.AnBstar output.AnB
