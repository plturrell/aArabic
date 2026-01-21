-- Proof tests

-- Simple propositions
theorem and_comm (p q : Prop) : p ∧ q → q ∧ p := by
  intro h
  exact ⟨h.2, h.1⟩

theorem or_comm (p q : Prop) : p ∨ q → q ∨ p := by
  intro h
  cases h with
  | inl hp => exact Or.inr hp
  | inr hq => exact Or.inl hq

-- Equality
theorem eq_symm {α : Type} (a b : α) : a = b → b = a := by
  intro h
  rw [h]

-- Natural number induction
theorem nat_add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.add_succ, ih]

-- Exists
theorem exists_example : ∃ n : Nat, n > 0 := by
  exists 1

