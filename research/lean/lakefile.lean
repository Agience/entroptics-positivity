import Lake
open Lake DSL

package «SignProblem» where
  -- keep proofs explicit; matches the paper's finite/discrete statements
  leanOptions := #[⟨`pp.unicode.fun, true⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.31.0"

@[default_target]
lean_lib «SignProblem» where
