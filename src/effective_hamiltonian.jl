"""
    mul_matrix_from_left(A, X)

```
     Λ  ┌───┐
(1)─1X2─1 A 2─(2)
     V  └─3─┘
          │
         (3)
```
"""
function mul_matrix_from_left(A::AbstractTensor3, X::AbstractMatrix)
    @tensoropt (p = 1, (l, r, x) = χ) mul_matrix_from_left[l, r, p] := X[l, x] * A[x, r, p]
end

"""
    mul_matrix_from_right(A, X)

```
    ┌───┐  Λ
(1)─1 A 2─1X2─(2)
    └─3─┘  V
      │
     (3)
```
"""
function mul_matrix_from_right(A::AbstractTensor3, X::AbstractMatrix)
    @tensoropt (p = 1, (l, r, x) = χ) mul_matrix_from_right[l, r, p] := A[l, x, p] * X[x, r]
end

"""
    mul_operator_onsite(A, O)

```
    ┌───┐
(1)─1 A 2─(2)
    └─3─┘
      │
     .2.
    ( O )
     `1'
      │
     (3)
```
"""
function mul_operator_onsite(A::AbstractTensor3, O::AbstractMatrix)
    @tensoropt ((p, x) = 1, (l, r) = χ) mul_operator_onsite[l, r, p] := A[l, r, x] * O[p, x]
end

"""
    mul_operator_with_left(A, O, AL)

```
 ┌────────┐  ┌───┐
┌1   AL   2──1 A 2─┐
│└────3───┘  └─3─┘ │
│     │        │   │
│    .3────────4.  │
│   (      O     ) │
│    `1────────2'  │
│     │        │   │
│┌────3───┐    │   │
└1conj(AL)2─┐  │  ┌┘
 └────────┘ │  │  │
(1)─────────┘ (3) └(2)
```
"""
function mul_operator_with_left(A::AbstractTensor3, O::AbstractTensor4, AL::AbstractTensor3)
    @tensoropt ((blp, p, klp, krp) = 1, (l, r, x, y) = χ) mul_operator_with_left[l, r, p] :=
        conj(AL)[y, l, blp] * AL[y, x, klp] * A[x, r, krp] * O[blp, p, klp, krp]
end

"""
    mul_operator_with_right(A, O, AR)

```
    ┌───┐  ┌────────┐
  ┌─1 A 2──1   AR   2┐
  │ └─3─┘  └───3────┘│
  │   │        │     │
  │  .3────────4.    │
  │ (     O      )   │
  │  `1────────2'    │
  │   │        │     │
  │   │    ┌───3────┐│
  └┐  │  ┌─1conj(AR)2┘
   │  │  │ └────────┘
(1)┘ (3) └───────────(2)
```
"""
function mul_operator_with_right(
    A::AbstractTensor3,
    O::AbstractTensor4,
    AR::AbstractTensor3,
)
    @tensoropt ((p, brp, klp, krp) = 1, (l, r, x, y) = χ) mul_operator_with_right[
        l,
        r,
        p,
    ] := A[l, x, klp] * AR[x, y, krp] * conj(AR)[r, y, brp] * O[p, brp, klp, krp]
end
