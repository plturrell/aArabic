-- Type system tests

-- Inductive types
inductive Color where
  | red
  | green
  | blue

-- Pattern matching
def colorToString (c : Color) : String :=
  match c with
  | Color.red => "red"
  | Color.green => "green"
  | Color.blue => "blue"

-- Option type
def safeDiv (a b : Nat) : Option Nat :=
  if b == 0 then none else some (a / b)

-- List operations
def listLength : List α → Nat
  | [] => 0
  | _ :: xs => 1 + listLength xs

-- Polymorphic identity
def id' (x : α) : α := x

-- Function composition
def compose (f : β → γ) (g : α → β) : α → γ :=
  fun x => f (g x)

