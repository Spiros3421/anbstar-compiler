"""
translator.py
Entry point: AnB* -> AnB

Usage:
    python translator.py <input.AnBstar> [output.AnB]

If no output file is given, result is printed to stdout.
"""

import sys
from anbstar_parser import parse_file
from compile import compile_protocol
from emit import emit


def translate_file(input_path: str, output_path: str = None) -> str:
    proto    = parse_file(input_path)
    compiled = compile_protocol(proto)
    result   = emit(compiled)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(result)
        print(f'Written to {output_path}')
    else:
        print(result)

    return result


def main():
    if len(sys.argv) < 2:
        print('Usage: python translator.py <input.AnBstar> [output.AnB]')
        sys.exit(1)
    translate_file(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else None)


if __name__ == '__main__':
    main()
