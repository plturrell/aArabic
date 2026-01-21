"""
Lean4 Standard Library support for nLeanProof.

This module provides implementations for core Lean4 standard library types:
- Nat: Natural numbers
- Int: Integers
- Bool: Booleans
- String: Strings
- List: Lists
- Option: Optional values
- Logic: Propositions and proofs
"""

from .nat import Nat, nat_add, nat_mul, nat_sub, nat_div, nat_mod
from .int import Int, int_add, int_neg, int_sub, int_mul
from .bool import lean_and, lean_or, lean_not, lean_ite
from .string import lean_string, string_append, string_length
from .list import LeanList, list_nil, list_cons, list_append, list_length
from .option import LeanOption, option_none, option_some, option_map
from .logic import Prop, True_, False_, And, Or, Not, Eq
