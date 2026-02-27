"""
emit.py
Pretty-prints a CompiledProtocol to OFMC AnB format.

Output structure:
    Protocol: <name>

    Types:
      Agent A, B;
      Number X1, Y1, ...;
      Function rk0, ck0, kdf_RKr, ...;

    Knowledge:
      A: A, B, g, rk0(A,B), ...;
      B: ...;
    where A != B

    Actions:

      A -> B: ...
      ...

    Goals:

      MsgA1 secret between A, B;
      ...

Note: format and ack are declared as Function (not Format) to avoid
OFMC Format-transparency issues.
"""

from compile import CompiledProtocol


def emit(cp: CompiledProtocol) -> str:
    lines = []

    # ------------------------------------------------------------------ #
    # Header                                                               #
    # ------------------------------------------------------------------ #
    lines.append(f'Protocol: {cp.name}')
    lines.append('')

    # ------------------------------------------------------------------ #
    # Types                                                                #
    # ------------------------------------------------------------------ #
    lines.append('Types:')
    lines.append('  Agent A, B;')
    if cp.numbers:
        lines.append(f'  Number {", ".join(cp.numbers)};')
    if cp.functions:
        lines.append(f'  Function {", ".join(cp.functions)};')
    if cp.formats:
        lines.append(f'  Format {", ".join(cp.formats)};')
    lines.append('')

    # ------------------------------------------------------------------ #
    # Knowledge                                                            #
    # ------------------------------------------------------------------ #
    lines.append('Knowledge:')
    for role, items in cp.knowledge:
        lines.append(f'  {role}: {", ".join(items)};')
    lines.append('where A != B')
    lines.append('')

    # ------------------------------------------------------------------ #
    # Actions                                                              #
    # ------------------------------------------------------------------ #
    lines.append('Actions:')
    lines.append('')
    for action in cp.actions:
        lines.append(f'  {action}')
    lines.append('')

    # ------------------------------------------------------------------ #
    # Goals                                                                #
    # ------------------------------------------------------------------ #
    lines.append('Goals:')
    lines.append('')
    for goal in cp.goals:
        lines.append(f'  {goal}')
    lines.append('')

    return '\n'.join(lines)
