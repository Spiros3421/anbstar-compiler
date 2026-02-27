"""
compile.py
σ/ρ compiler: takes a parsed Protocol and produces a CompiledProtocol.

Translation rules (from TranslatorTest.txt section 4):
  - rho  : let env     (local bindings, reset after each New State block)
  - sigma: state env   (state variables, updated by New State blocks)

  Atom(id) -> rho[id]  if id in rho
            -> sigma[id] if id in sigma
            -> id        otherwise
  App(f, args) -> f(translate(a) for a in args)
  Enc(m, k)    -> {| translate(m) |} translate(k)
  Tup(ts)      -> ", ".join(translate(t) for t in ts)

NewStateBlock:
  - Compute all new values simultaneously from current (sigma, rho)
  - Update sigma
  - Reset rho to {}

Let:
  - rho[name] = translate(term, sigma, rho)   (sequential; sees previous lets)

Send:
  - Emit "src -> dst: translate(payload, sigma, rho)"
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from anbstar_parser import (
    Protocol, Term, Atom, App, Enc, Tup,
    Fresh, Send, Let, NewStateBlock, Repeat,
    StateDecl, KnowledgeDecl,
)


# ---------------------------------------------------------------------------
# Compiled output (handed to emit.py)
# ---------------------------------------------------------------------------

@dataclass
class CompiledProtocol:
    name:      str
    numbers:   List[str]           # Number section (fresh vars + msg vars)
    functions: List[str]           # Function section (state funcs + user funcs)
    formats: List[str]
    knowledge: List[Tuple[str, List[str]]]  # [(role, [item_str, ...]), ...]
    actions:   List[str]           # translated "A -> B: ..." lines
    goals:     List[str]           # raw goal strings


# ---------------------------------------------------------------------------
# Built-in / reserved function names — excluded from the Types: Function list
# ---------------------------------------------------------------------------
# exp  — built-in OFMC DH constructor
# g    — public DH generator constant (Atom, not a function anyway)
_BUILTIN_FUNCS = {'exp'}

# Atoms whose name ends in '1' (e.g. Msg1, Key1) inside a Repeat body are
# treated as "indexed templates": iteration i renames Msg1 → Msg{i}.
_INDEXED_ATOM_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)1$')

# Format-function symbols:
#   FORMAT_INDEXED — renamed per Repeat iteration (format → format1, format2, …)
#   FORMAT_STABLE  — declared as Format once, never renamed (ack stays ack)
#   FORMAT_SYMBOLS — union; all are emitted as "Format" not "Function"
FORMAT_INDEXED = {'format'}
FORMAT_STABLE  = {'ack'}
FORMAT_SYMBOLS = FORMAT_INDEXED | FORMAT_STABLE


def _format_funcs_in_term(term: Term) -> set:
    """Return FORMAT_INDEXED names used as App.func anywhere in *term*."""
    found: set = set()
    if isinstance(term, App):
        if term.func in FORMAT_INDEXED:
            found.add(term.func)
        for a in term.args:
            found |= _format_funcs_in_term(a)
    elif isinstance(term, Enc):
        found |= _format_funcs_in_term(term.msg)
        found |= _format_funcs_in_term(term.key)
    elif isinstance(term, Tup):
        for t in term.terms:
            found |= _format_funcs_in_term(t)
    return found


def _format_funcs_in_stmts(stmts) -> set:
    """Recursively collect FORMAT_INDEXED used as App.func in a stmt list
    (including nested Repeat bodies). Used only for top-level detection."""
    found: set = set()
    for stmt in stmts:
        if isinstance(stmt, Send):
            found |= _format_funcs_in_term(stmt.payload)
        elif isinstance(stmt, Let):
            found |= _format_funcs_in_term(stmt.term)
        elif isinstance(stmt, Repeat):
            found |= _format_funcs_in_stmts(stmt.body)
    return found


def _format_funcs_in_direct_stmts(stmts) -> set:
    """Collect FORMAT_INDEXED used as App.func in *direct* Send/Let stmts only.
    Does NOT recurse into nested Repeat blocks — those are handled when
    _prescan_repeat / _compile_repeat recurse into the inner Repeat.
    This prevents double-counting the counter for nested Repeat levels."""
    found: set = set()
    for stmt in stmts:
        if isinstance(stmt, Send):
            found |= _format_funcs_in_term(stmt.payload)
        elif isinstance(stmt, Let):
            found |= _format_funcs_in_term(stmt.term)
    return found


# ---------------------------------------------------------------------------
# Term translation
# ---------------------------------------------------------------------------

def translate(term: Term,
              sigma: Dict[str, str],
              rho:   Dict[str, str]) -> str:
    """Translate an AnB* term to an AnB string using current σ and ρ."""

    if isinstance(term, Atom):
        name = term.name
        # Resolution order: rho first, then sigma, then identity
        if name in rho:
            return rho[name]
        if name in sigma:
            return sigma[name]
        return name

    if isinstance(term, App):
        targs = [translate(a, sigma, rho) for a in term.args]
        return f"{term.func}({', '.join(targs)})"

    if isinstance(term, Enc):
        tmsg = translate(term.msg, sigma, rho)
        tkey = translate(term.key, sigma, rho)
        return f"{{| {tmsg} |}} {tkey}"

    if isinstance(term, Tup):
        return ', '.join(translate(t, sigma, rho) for t in term.terms)

    raise TypeError(f'Unknown term type: {type(term).__name__}')


def _resolve(name: str, sigma: Dict[str, str], rho: Dict[str, str]) -> str:
    """Resolve a plain identifier against rho then sigma then identity."""
    if name in rho:
        return rho[name]
    if name in sigma:
        return sigma[name]
    return name


# ---------------------------------------------------------------------------
# Function symbol collector
# ---------------------------------------------------------------------------

def _collect_funcs(term: Term, funcs: List[str]) -> None:
    """Walk a term and append function names (in order of first appearance),
    skipping built-ins."""
    if isinstance(term, App):
        if term.func not in _BUILTIN_FUNCS and term.func not in funcs:
            funcs.append(term.func)
        for a in term.args:
            _collect_funcs(a, funcs)
    elif isinstance(term, Enc):
        _collect_funcs(term.msg, funcs)
        _collect_funcs(term.key, funcs)
    elif isinstance(term, Tup):
        for t in term.terms:
            _collect_funcs(t, funcs)
    # Atom: no function name to collect


# ---------------------------------------------------------------------------
# Recursive function-symbol collector (handles Repeat bodies)
# ---------------------------------------------------------------------------

def _collect_all_funcs(stmts: List, out: List[str]) -> None:
    """Walk all statements (including Repeat bodies) and collect function names."""
    for stmt in stmts:
        if isinstance(stmt, Send):
            _collect_funcs(stmt.payload, out)
        elif isinstance(stmt, Let):
            _collect_funcs(stmt.term, out)
        elif isinstance(stmt, Repeat):
            _collect_all_funcs(stmt.body, out)


# ---------------------------------------------------------------------------
# Indexed-atom collector (for Repeat renaming)
# ---------------------------------------------------------------------------

def _atoms_in_term(term: Term, out: set) -> None:
    """Collect all Atom names from a term."""
    if isinstance(term, Atom):
        out.add(term.name)
    elif isinstance(term, App):
        for a in term.args:
            _atoms_in_term(a, out)
    elif isinstance(term, Enc):
        _atoms_in_term(term.msg, out)
        _atoms_in_term(term.key, out)
    elif isinstance(term, Tup):
        for t in term.terms:
            _atoms_in_term(t, out)


def _collect_indexed_atoms(stmts: List, body_fresh: List[str]) -> List[str]:
    """Return atoms matching <name>1 from Send/Let stmts, excluding fresh vars.

    These are "indexed templates": Msg1 in iteration i becomes Msg{i}.
    Atoms that are iteration-suffixed forms of a fresh var (e.g. n1 when n
    is fresh) are also excluded to avoid conflicts.
    """
    fresh_indexed = {f'{v}1' for v in body_fresh}
    found: set = set()
    for stmt in stmts:
        if isinstance(stmt, Send):
            _atoms_in_term(stmt.payload, found)
        elif isinstance(stmt, Let):
            _atoms_in_term(stmt.term, found)
    result = []
    for name in sorted(found):
        if (_INDEXED_ATOM_RE.match(name)
                and name not in body_fresh
                and name not in fresh_indexed):
            result.append(name)
    return result


# ---------------------------------------------------------------------------
# Term translation with a per-iteration rename map (for Repeat unfolding)
# ---------------------------------------------------------------------------

def _translate_renamed(term: Term,
                        sigma: Dict[str, str],
                        rho:   Dict[str, str],
                        rename: Dict[str, str]) -> str:
    """Like translate() but applies `rename` before the normal σ/ρ lookup.

    Used to redirect bare Fresh variable names to their iteration-suffixed
    counterparts (e.g.  n  →  n_2  inside iteration 2 of a Repeat block).
    """
    if isinstance(term, Atom):
        name = rename.get(term.name, term.name)
        if name in rho:
            return rho[name]
        if name in sigma:
            return sigma[name]
        return name

    if isinstance(term, App):
        func_name = rename.get(term.func, term.func)
        targs = [_translate_renamed(a, sigma, rho, rename) for a in term.args]
        return f"{func_name}({', '.join(targs)})"

    if isinstance(term, Enc):
        tmsg = _translate_renamed(term.msg, sigma, rho, rename)
        tkey = _translate_renamed(term.key, sigma, rho, rename)
        return f"{{| {tmsg} |}} {tkey}"

    if isinstance(term, Tup):
        return ', '.join(_translate_renamed(t, sigma, rho, rename) for t in term.terms)

    raise TypeError(f'Unknown term type: {type(term).__name__}')


# ---------------------------------------------------------------------------
# Pre-scan for Types: Number and global counters
# ---------------------------------------------------------------------------

def _prescan_repeat(repeat: Repeat,
                    fresh_vars: List[str],
                    counters: Dict[str, int]) -> None:
    """Recursively pre-scan a Repeat (including nested Repeats) to:
      - add renamed fresh/indexed-atom vars to fresh_vars (Types: Number)
      - accumulate global counters for indexed atoms and FORMAT_INDEXED funcs

    counters is shared across all recursive calls so the index never resets
    when entering a new outer iteration or a nested Repeat.
    """
    body_fresh      = [s.var for s in repeat.body if isinstance(s, Fresh)]
    indexed_atoms   = _collect_indexed_atoms(repeat.body, body_fresh)
    body_fmt_funcs  = _format_funcs_in_direct_stmts(repeat.body)

    for i in range(1, repeat.count + 1):
        # Fresh vars keep iteration-local names (var1, var2, …)
        for var in body_fresh:
            renamed = f'{var}{i}'
            if renamed not in fresh_vars:
                fresh_vars.append(renamed)

        # Indexed atoms use the global counter → Msg1, Msg2, … across all levels
        for atom in indexed_atoms:
            base = atom[:-1]
            counters[atom] = counters.get(atom, 0) + 1
            renamed = f'{base}{counters[atom]}'
            if renamed not in fresh_vars:
                fresh_vars.append(renamed)

        # FORMAT_INDEXED funcs: just count (names are built later)
        for fmt in body_fmt_funcs:
            counters[fmt] = counters.get(fmt, 0) + 1

        # Recurse into nested Repeats in body order
        for stmt in repeat.body:
            if isinstance(stmt, Repeat):
                _prescan_repeat(stmt, fresh_vars, counters)


# ---------------------------------------------------------------------------
# Bounded unfolding of a Repeat block
# ---------------------------------------------------------------------------

def _compile_repeat(repeat: Repeat,
                    sigma: Dict[str, str],
                    compiled_actions: List[str],
                    outer_fresh: List[str],
                    counters: Dict[str, int]) -> None:
    """Unfold a Repeat block into K sequential copies.

    Semantics implemented:
      - σ carries forward across iterations (mutates the caller's dict).
      - ρ is reset to {} at the start of every iteration.
      - Fresh vars are renamed to  <var>_<i>  for iteration i.
      - Illegal variable reuse raises ValueError.
    """
    body_fresh = [s.var for s in repeat.body if isinstance(s, Fresh)]
    indexed_atoms = _collect_indexed_atoms(repeat.body, body_fresh)
    body_format_funcs = _format_funcs_in_direct_stmts(repeat.body)

    # ---- Illegal-reuse checks ----------------------------------------
    conflicts_outer = set(body_fresh) & set(outer_fresh)
    if conflicts_outer:
        raise ValueError(
            f"Illegal variable reuse in Repeat body: "
            f"Fresh vars {sorted(conflicts_outer)} are already declared "
            f"in the outer scope (would create duplicate Types: Number entries)."
        )

    conflicts_sigma = set(body_fresh) & set(sigma)
    if conflicts_sigma:
        raise ValueError(
            f"Illegal variable reuse in Repeat body: "
            f"Fresh vars {sorted(conflicts_sigma)} conflict with state "
            f"variables (sigma) of the same name."
        )

    # ---- Bounded unfolding -------------------------------------------
    for i in range(1, repeat.count + 1):
        # Reset ρ at the start of each iteration
        rho_iter: Dict[str, str] = {}

        # Fresh vars: still use local iteration index (var1, var2, …)
        rename = {var: f'{var}{i}' for var in body_fresh}
        # Indexed atoms: global counter → never resets across nested Repeats
        for atom in indexed_atoms:
            base = atom[:-1]
            counters[atom] = counters.get(atom, 0) + 1
            rename[atom] = f'{base}{counters[atom]}'
        # FORMAT_INDEXED funcs: global counter → format1, format2, … continuously
        for fmt in body_format_funcs:
            counters[fmt] = counters.get(fmt, 0) + 1
            rename[fmt] = f'{fmt}{counters[fmt]}'

        for stmt in repeat.body:

            if isinstance(stmt, Fresh):
                pass   # already added to Types: Number during pre-scan

            elif isinstance(stmt, Let):
                rho_iter[stmt.name] = _translate_renamed(
                    stmt.term, sigma, rho_iter, rename)

            elif isinstance(stmt, NewStateBlock):
                # Parallel update: resolve all RHS against current (σ, ρ, rename)
                new_vals = {}
                for lhs, rhs in stmt.updates:
                    r = rename.get(rhs, rhs)
                    new_vals[lhs] = _resolve(r, sigma, rho_iter)
                sigma.update(new_vals)
                rho_iter = {}   # NewStateBlock also resets ρ within an iteration

            elif isinstance(stmt, Send):
                payload_str = _translate_renamed(
                    stmt.payload, sigma, rho_iter, rename)
                compiled_actions.append(f'{stmt.src} -> {stmt.dst}: {payload_str}')

            elif isinstance(stmt, Repeat):
                # Nested Repeat: recurse.  outer_fresh for the inner block
                # is outer_fresh ∪ body_fresh (with iteration suffix).
                inner_outer = outer_fresh + [f'{v}{i}' for v in body_fresh]
                _compile_repeat(stmt, sigma, compiled_actions, inner_outer, counters)


# ---------------------------------------------------------------------------
# Main compiler
# ---------------------------------------------------------------------------

def compile_protocol(proto: Protocol) -> CompiledProtocol:

    # ------------------------------------------------------------------
    # 1. Initialise sigma0 from State block
    #    S <- new  =>  sigma0[S] = s0(A,B)   and  declare s0 as Function
    # ------------------------------------------------------------------
    sigma0: Dict[str, str] = {}
    state_funcs: List[str] = []

    for sd in proto.state:
        sym = sd.name.lower() + '0'
        sigma0[sd.name] = f'{sym}(A,B)'
        if sym not in state_funcs:
            state_funcs.append(sym)

    # ------------------------------------------------------------------
    # 2. Collect fresh vars (from Fresh stmts) and message vars (from Goals)
    #    Fresh vars inside Repeat blocks are renamed  var_1 … var_K  so
    #    each iteration gets a distinct Number entry.
    # ------------------------------------------------------------------
    # outer_fresh: Fresh vars declared directly in the Actions block
    outer_fresh: List[str] = []
    fresh_vars:  List[str] = []
    # prescan_counters: global counter shared across all Repeat pre-scans,
    # tracks total uses of each indexed atom and FORMAT_INDEXED func symbol.
    prescan_counters: Dict[str, int] = {}

    for stmt in proto.actions:
        if isinstance(stmt, Fresh):
            if stmt.var not in outer_fresh:
                outer_fresh.append(stmt.var)
            if stmt.var not in fresh_vars:
                fresh_vars.append(stmt.var)
        elif isinstance(stmt, Repeat):
            _prescan_repeat(stmt, fresh_vars, prescan_counters)

    msg_vars: List[str] = []
    for goal in proto.goals:
        m = re.match(r'^(\w+)\s+secret', goal)
        if m:
            v = m.group(1)
            if v not in msg_vars:
                msg_vars.append(v)
        mm = re.search(r'authenticates\s+\w+\s+on\s+(\w+)', goal)
        if mm:
            v = mm.group(1)
            if v not in msg_vars:
                msg_vars.append(v)

    numbers = fresh_vars + [v for v in msg_vars if v not in fresh_vars]

    # ------------------------------------------------------------------
    # 3. Collect user-defined function symbols (in order of first appearance)
    #    Scan Send payloads and Let binding terms, including Repeat bodies.
    # ------------------------------------------------------------------
    user_funcs: List[str] = []
    _collect_all_funcs(proto.actions, user_funcs)

    # Full function list: state symbols first, then user-defined
    all_funcs = list(state_funcs)
    for f in user_funcs:
        if f not in all_funcs:
            all_funcs.append(f)

    # Build repeat_formats from prescan_counters: e.g. format appeared 6 times
    # total (across all Repeat iterations and nesting levels) → format1…format6.
    repeat_formats: List[str] = []
    for fmt in sorted(f for f in prescan_counters if f in FORMAT_INDEXED):
        for i in range(1, prescan_counters[fmt] + 1):
            repeat_formats.append(f'{fmt}{i}')

    # FORMAT_INDEXED symbols that appear *only* inside Repeat bodies must not be
    # emitted unrenamed — only their per-iteration forms are valid.
    fmt_only_in_repeats = set(prescan_counters) & FORMAT_INDEXED
    for stmt in proto.actions:
        if not isinstance(stmt, Repeat):
            if isinstance(stmt, Send):
                fmt_only_in_repeats -= _format_funcs_in_term(stmt.payload)
            elif isinstance(stmt, Let):
                fmt_only_in_repeats -= _format_funcs_in_term(stmt.term)

    formats = (
        [f for f in all_funcs if f in FORMAT_SYMBOLS and f not in fmt_only_in_repeats]
        + repeat_formats
    )
    functions = [f for f in all_funcs if f not in FORMAT_SYMBOLS]

    # ------------------------------------------------------------------
    # 4. Translate Knowledge block using sigma0
    #    Replace state variable names with their sigma0 bindings;
    #    all other items pass through unchanged.
    # ------------------------------------------------------------------
    compiled_knowledge: List[Tuple[str, List[str]]] = []
    for kd in proto.knowledge:
        items_out = []
        for item in kd.items:
            if item in sigma0:
                items_out.append(sigma0[item])
            else:
                items_out.append(item)
        compiled_knowledge.append((kd.role, items_out))

    # ------------------------------------------------------------------
    # 5. Compile actions using the σ/ρ engine
    # ------------------------------------------------------------------
    sigma: Dict[str, str] = dict(sigma0)   # evolving state env
    rho:   Dict[str, str] = {}             # let env, reset at each NewStateBlock

    compiled_actions: List[str] = []
    # compile_counters mirrors prescan_counters but starts fresh so the two
    # passes produce identical symbol names in the same traversal order.
    compile_counters: Dict[str, int] = {}

    for stmt in proto.actions:

        if isinstance(stmt, Fresh):
            pass  # declared in Types: Number

        elif isinstance(stmt, Let):
            # Sequential: this binding sees all previous rho entries
            rho[stmt.name] = translate(stmt.term, sigma, rho)

        elif isinstance(stmt, NewStateBlock):
            # Parallel update: compute all new values from current (sigma, rho)
            new_vals = {lhs: _resolve(rhs, sigma, rho)
                        for lhs, rhs in stmt.updates}
            sigma.update(new_vals)
            rho = {}   # reset let env after state transition

        elif isinstance(stmt, Send):
            payload_str = translate(stmt.payload, sigma, rho)
            compiled_actions.append(f'{stmt.src} -> {stmt.dst}: {payload_str}')

        elif isinstance(stmt, Repeat):
            _compile_repeat(stmt, sigma, compiled_actions, outer_fresh, compile_counters)

    # ------------------------------------------------------------------
    # 6. Assemble result
    # ------------------------------------------------------------------
    return CompiledProtocol(
        name=proto.name,
        numbers=numbers,
        functions=functions,
        formats=formats,
        knowledge=compiled_knowledge,
        actions=compiled_actions,
        goals=proto.goals,
    )
