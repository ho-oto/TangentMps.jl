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

    let (_, _l, _r, _p, _x) = ntuple(x -> nothing, 5)
        @tensor _[_l, _r, _p] := X[_l, _x] * A[_x, _r, _p]
    end
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

    let (_, _l, _r, _p, _x) = ntuple(x -> nothing, 5)
        @tensor _[_l, _r, _p] := A[_l, _x, _p] * X[_x, _r]
    end
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

    let (_, _l, _r, _p, _x) = ntuple(x -> nothing, 5)
        @tensor _[_l, _r, _p] := A[_l, _r, _x] * O[_p, _x]
    end
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

    let (_, _l, _r, _blp, _p, _klp, _krp, _x, _y) = ntuple(x -> nothing, 9)
        @tensoropt !(_blp, _p, _klp, _krp) _[_l, _r, _p] :=
            conj(AL)[_y, _l, _blp] *
            AL[_y, _x, _klp] *
            A[_x, _r, _krp] *
            O[_blp, _p, _klp, _krp]
    end

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

    let (_, _l, _r, _p, _brp, _klp, _krp, _x, _y) = ntuple(x -> nothing, 9)
        @tensoropt !(_p, _brp, _klp, _krp) _[_l, _r, _p] :=
            A[_l, _x, _klp] *
            AR[_x, _y, _krp] *
            conj(AR)[_r, _y, _brp] *
            O[_p, _brp, _klp, _krp]
    end
end
