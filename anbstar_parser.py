"""
anbstar_parser.py
Tokenises and parses an AnB* source file into a Protocol AST.

DH terms are written as exp(g, X) or exp(exp(g, X), Y) — plain function calls.
No special '^' handling required.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union


# ---------------------------------------------------------------------------
# AST — Terms
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    """Plain identifier: agent name, variable, constant."""
    name: str

@dataclass
class App:
    """Function application: f(t1, ..., tn).
    Also covers DH: exp(g, X), exp(exp(g, X), Y)."""
    func: str
    args: List['Term']

@dataclass
class Enc:
    """Symmetric encryption: {| msg |}_key"""
    msg: 'Term'
    key: 'Term'

@dataclass
class Tup:
    """Comma-separated payload tuple (only at the top level of a Send)."""
    terms: List['Term']

Term = Union[Atom, App, Enc, Tup]


# ---------------------------------------------------------------------------
# AST — Statements
# ---------------------------------------------------------------------------

@dataclass
class Fresh:
    """Role: new var"""
    role: str
    var: str

@dataclass
class Send:
    """src -> dst : payload"""
    src: str
    dst: str
    payload: Term

@dataclass
class Let:
    """Single let binding: name = term  (sequential; each sees previous bindings)"""
    name: str
    term: Term

@dataclass
class NewStateBlock:
    """New State block: simultaneous update of state variables.
    All RHS are resolved against the sigma/rho *before* any update."""
    updates: List[Tuple[str, str]]   # [(lhs_ident, rhs_ident), ...]

@dataclass
class Repeat:
    """Repeat K: body EndRepeat — bounded unfolding, executed K times.

    Semantics:
      - Execute body exactly K times (iterations 1 … K).
      - σ (state) updates carry forward across iterations.
      - ρ (let-bindings) is reset to {} at the start of every iteration.
      - Fresh vars inside are renamed to var_1 … var_K to prevent reuse.
    """
    count: int
    body: List['Stmt']


Stmt = Union[Fresh, Send, Let, NewStateBlock, Repeat]


# ---------------------------------------------------------------------------
# AST — Top-level blocks
# ---------------------------------------------------------------------------

@dataclass
class StateDecl:
    """State: varname <- new"""
    name: str
    init: str   # always 'new' for now

@dataclass
class KnowledgeDecl:
    """Knowledge: role : item1, item2, ..."""
    role: str
    items: List[str]

@dataclass
class Protocol:
    name: str
    state:     List[StateDecl]     = field(default_factory=list)
    knowledge: List[KnowledgeDecl] = field(default_factory=list)
    actions:   List[Stmt]          = field(default_factory=list)
    goals:     List[str]           = field(default_factory=list)  # raw strings


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r'(?P<COMMENT>#[^\n]*)'        # # ... to end of line
    r'|(?P<LENC>\{\|)'             # {|
    r'|(?P<RENC>\|}_)'             # |}_   (must precede any lone | or })
    r'|(?P<ARROW>->)'              # ->    (must precede lone -)
    r'|(?P<LARROW><-)'             # <-
    r'|(?P<LPAREN>\()'
    r'|(?P<RPAREN>\))'
    r'|(?P<COMMA>,)'
    r'|(?P<COLON>:)'
    r'|(?P<SEMI>;)'
    r'|(?P<EQ>=)'
    r'|(?P<NUMBER>[0-9]+)'  # new for repeat
    r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_']*)"
    r'|(?P<NEWLINE>\n)'
    r'|(?P<SKIP>[ \t\r]+)'
)


def tokenise(text: str) -> List[Tuple[str, str]]:
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind in ('SKIP', 'COMMENT'):
            continue
        tokens.append((kind, m.group()))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Keywords that start a new top-level block
_BLOCK_KW = {'State', 'Knowledge', 'Actions', 'Goals', 'Let', 'New', 'Repeat'}
# Keywords that end the current sub-block
_STOP_KW  = {'State', 'Knowledge', 'Actions', 'Goals', 'Let', 'New', 'Repeat'}

# Punctuation tokens that should not be preceded by a space
_NO_SPACE_BEFORE = {'COMMA', 'SEMI', 'RPAREN', 'COLON'}
_NO_SPACE_AFTER  = {'LPAREN'}

def _join_tokens(parts: List[Tuple[str, str]]) -> str:
    """Join token list into a string with sensible spacing (no space before , ; ) :)."""
    result = ''
    for i, (kind, val) in enumerate(parts):
        if i == 0:
            result = val
        elif kind in _NO_SPACE_BEFORE or parts[i-1][0] in _NO_SPACE_AFTER:
            result += val
        else:
            result += ' ' + val
    return result.strip()


class Parser:
    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tok = tokens
        self.pos = 0

    # ------------------------------------------------------------------ #
    # Low-level helpers                                                    #
    # ------------------------------------------------------------------ #

    def _skip_nl(self):
        while self.pos < len(self.tok) and self.tok[self.pos][0] == 'NEWLINE':
            self.pos += 1

    def peek(self) -> Optional[Tuple[str, str]]:
        self._skip_nl()
        return self.tok[self.pos] if self.pos < len(self.tok) else None

    def eat(self, kind: str = None) -> Tuple[str, str]:
        self._skip_nl()
        if self.pos >= len(self.tok):
            raise SyntaxError(f'Unexpected EOF (expected {kind!r})')
        tok = self.tok[self.pos]
        if kind and tok[0] != kind:
            raise SyntaxError(f'Expected {kind!r}, got {tok!r} at token #{self.pos}')
        self.pos += 1
        return tok

    def at_end(self) -> bool:
        p = self.pos
        while p < len(self.tok) and self.tok[p][0] == 'NEWLINE':
            p += 1
        return p >= len(self.tok)

    def next_is(self, kind: str, value: str = None) -> bool:
        t = self.peek()
        if t is None:
            return False
        return t[0] == kind and (value is None or t[1] == value)

    def _at_stop(self, extra=()) -> bool:
        """True if the next non-NL token is a block-stopping keyword."""
        t = self.peek()
        return (t is not None and t[0] == 'IDENT'
                and t[1] in (_STOP_KW | set(extra)))

    # ------------------------------------------------------------------ #
    # Top-level                                                            #
    # ------------------------------------------------------------------ #

    def parse(self) -> Protocol:
        proto = Protocol(name='')
        self._skip_nl()

        if self.next_is('IDENT', 'Protocol'):
            self.eat(); self.eat('COLON')
            proto.name = self.eat('IDENT')[1]

        while not self.at_end():
            t = self.peek()
            if t is None:
                break
            if t[0] != 'IDENT':
                self.pos += 1
                continue

            kw = t[1]
            if kw == 'State':
                self.eat(); self.eat('COLON')
                proto.state = self._parse_state()
            elif kw == 'Knowledge':
                self.eat(); self.eat('COLON')
                proto.knowledge = self._parse_knowledge()
            elif kw == 'Actions':
                self.eat(); self.eat('COLON')
                proto.actions = self._parse_actions()
            elif kw == 'Goals':
                self.eat(); self.eat('COLON')
                proto.goals = self._parse_goals()
            else:
                self.eat()   # skip unknown top-level token

        return proto

    # ------------------------------------------------------------------ #
    # State block                                                          #
    # ------------------------------------------------------------------ #

    def _parse_state(self) -> List[StateDecl]:
        decls = []
        while not self.at_end() and not self._at_stop():
            t = self.peek()
            if t is None or t[0] != 'IDENT':
                break
            name = self.eat('IDENT')[1]
            self.eat('LARROW')
            t2 = self.peek()
            if t2 and t2[0] == 'IDENT' and t2[1] == 'new':
                self.eat()
                decls.append(StateDecl(name=name, init='new'))
            else:
                # Term init — not supported yet; treat as 'new'
                decls.append(StateDecl(name=name, init='new'))
        return decls

    # ------------------------------------------------------------------ #
    # Knowledge block                                                      #
    # ------------------------------------------------------------------ #

    def _parse_knowledge(self) -> List[KnowledgeDecl]:
        decls = []
        stop = {'Actions', 'Goals', 'State'}
        while not self.at_end():
            t = self.peek()
            if t is None or (t[0] == 'IDENT' and t[1] in stop):
                break
            if t[0] != 'IDENT':
                self.pos += 1
                continue
            role = self.eat('IDENT')[1]
            self.eat('COLON')
            items = [self.eat('IDENT')[1]]
            while self.next_is('COMMA'):
                self.eat('COMMA')
                items.append(self.eat('IDENT')[1])
            decls.append(KnowledgeDecl(role=role, items=items))
        return decls

    # ------------------------------------------------------------------ #
    # Actions block                                                        #
    # ------------------------------------------------------------------ #

    def _parse_actions(self) -> List[Stmt]:
        return self._parse_action_stmts(stop_at={'Goals', 'EndRepeat'})

    def _parse_action_stmts(self, stop_at: set) -> List[Stmt]:
        """Parse a sequence of action statements until a keyword in stop_at.

        If 'EndRepeat' is in stop_at and is encountered, it is consumed.
        All other stop keywords are left in the token stream for the caller.
        """
        stmts = []
        while not self.at_end():
            t = self.peek()
            if t is None:
                break
            if t[0] == 'IDENT' and t[1] in stop_at:
                if t[1] == 'EndRepeat':
                    self.eat()   # consume terminator
                break

            # --- Let block ---
            if t[0] == 'IDENT' and t[1] == 'Let':
                self.eat(); self.eat('COLON')
                for name, term in self._parse_let_bindings():
                    stmts.append(Let(name=name, term=term))

            # --- New State block ---
            elif t[0] == 'IDENT' and t[1] == 'New':
                self.eat()
                t2 = self.peek()
                if t2 and t2[0] == 'IDENT' and t2[1] == 'State':
                    self.eat(); self.eat('COLON')
                    updates = self._parse_newstate_bindings()
                    stmts.append(NewStateBlock(updates=updates))

            # --- Repeat block ---
            elif t[0] == 'IDENT' and t[1] == 'Repeat':
                stmts.append(self._parse_repeat())

            # --- Fresh or Send ---
            elif t[0] == 'IDENT':
                saved = self.pos
                role = self.eat('IDENT')[1]

                if self.next_is('COLON'):
                    self.eat('COLON')
                    t2 = self.peek()
                    # Fresh: Role: new var
                    if t2 and t2[0] == 'IDENT' and t2[1] == 'new':
                        self.eat()
                        var = self.eat('IDENT')[1]
                        stmts.append(Fresh(role=role, var=var))
                    else:
                        # Not fresh — backtrack, parse as send
                        self.pos = saved
                        stmts.append(self._parse_send())

                elif self.next_is('ARROW'):
                    self.pos = saved
                    stmts.append(self._parse_send())
                else:
                    # Unknown line — skip token
                    pass
            else:
                self.pos += 1

        return stmts

    def _parse_repeat(self) -> Repeat:
        """Parse:  Repeat <NUMBER> : <body> EndRepeat"""
        self.eat('IDENT')                      # 'Repeat'
        count = int(self.eat('NUMBER')[1])
        self.eat('COLON')
        body = self._parse_action_stmts(stop_at={'EndRepeat'})
        return Repeat(count=count, body=body)

    def _parse_send(self) -> Send:
        src = self.eat('IDENT')[1]
        self.eat('ARROW')
        dst = self.eat('IDENT')[1]
        self.eat('COLON')
        payload = self._parse_payload()
        return Send(src=src, dst=dst, payload=payload)

    def _parse_payload(self) -> Term:
        """Top-level: comma-separated tuple."""
        terms = [self._parse_term()]
        while self.next_is('COMMA'):
            self.eat('COMMA')
            terms.append(self._parse_term())
        return terms[0] if len(terms) == 1 else Tup(terms=terms)

    # ------------------------------------------------------------------ #
    # Let bindings                                                         #
    # ------------------------------------------------------------------ #

    def _parse_let_bindings(self) -> List[Tuple[str, 'Term']]:
        bindings = []
        stop = {'New', 'Let', 'Goals', 'Actions', 'State', 'Repeat', 'EndRepeat'}
        while not self.at_end():
            t = self.peek()
            if t is None:
                break
            if t[0] == 'IDENT' and t[1] in stop:
                break
            if t[0] != 'IDENT':
                break
            saved = self.pos
            name = self.eat('IDENT')[1]
            if self.next_is('EQ'):
                self.eat('EQ')
                term = self._parse_term()
                bindings.append((name, term))
            else:
                self.pos = saved
                break
        return bindings

    # ------------------------------------------------------------------ #
    # New State bindings                                                   #
    # ------------------------------------------------------------------ #

    def _parse_newstate_bindings(self) -> List[Tuple[str, str]]:
        """Returns [(lhs_ident, rhs_ident), ...].
        Both sides are plain identifiers — no full term parsing needed."""
        updates = []
        stop = {'Let', 'New', 'Goals', 'Actions', 'State', 'Repeat', 'EndRepeat'}
        while not self.at_end():
            t = self.peek()
            if t is None:
                break
            if t[0] == 'IDENT' and t[1] in stop:
                break
            if t[0] != 'IDENT':
                break
            saved = self.pos
            lhs = self.eat('IDENT')[1]
            if self.next_is('EQ'):
                self.eat('EQ')
                rhs = self.eat('IDENT')[1]
                updates.append((lhs, rhs))
            else:
                self.pos = saved
                break
        return updates

    # ------------------------------------------------------------------ #
    # Goals block (raw strings)                                            #
    # ------------------------------------------------------------------ #

    def _parse_goals(self) -> List[str]:
        goals = []
        parts: List[Tuple[str, str]] = []
        while self.pos < len(self.tok):
            kind, val = self.tok[self.pos]
            if kind == 'NEWLINE':
                if parts:
                    line = _join_tokens(parts)
                    if line:
                        goals.append(line)
                    parts = []
            else:
                parts.append((kind, val))
            self.pos += 1
        if parts:
            line = _join_tokens(parts)
            if line:
                goals.append(line)
        return goals

    # ------------------------------------------------------------------ #
    # Term parser                                                          #
    # ------------------------------------------------------------------ #

    def _parse_term(self) -> Term:
        t = self.peek()
        if t is None:
            raise SyntaxError('Unexpected EOF while parsing term')

        # {| msg |}_key
        if t[0] == 'LENC':
            self.eat('LENC')
            msg = self._parse_term()
            self.eat('RENC')
            key = self._parse_term()
            return Enc(msg=msg, key=key)

        if t[0] != 'IDENT':
            raise SyntaxError(f'Expected term atom or enc, got {t!r}')

        name = self.eat('IDENT')[1]

        # Function application: name(args...)
        if self.next_is('LPAREN'):
            self.eat('LPAREN')
            args = []
            if not self.next_is('RPAREN'):
                args.append(self._parse_term())
                while self.next_is('COMMA'):
                    self.eat('COMMA')
                    args.append(self._parse_term())
            self.eat('RPAREN')
            return App(func=name, args=args)

        return Atom(name=name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(path: str) -> Protocol:
    with open(path, 'r') as f:
        text = f.read()
    tokens = tokenise(text)
    return Parser(tokens).parse()


def parse_string(text: str) -> Protocol:
    tokens = tokenise(text)
    return Parser(tokens).parse()
