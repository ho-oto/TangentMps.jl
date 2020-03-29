"""
    transfer_from_left(X, [O,] A, [B=A])

```
   Λ
  ╱ ╲  ┌───────┐
┌┤ X ├─┤   A   ├──
│ ╲ ╱  └───┬───┘
│  V       │
│         .┴.
│        ( O )
│         `┬'
│          │
│      ┌───┴───┐
└──────│conj(B)│─┐
       └───────┘ │
─────────────────┘
```
"""
function transfer_from_left(X::AbstractMatrix, A::AbstractTensor3, B::AbstractTensor3 = A)
    @tensoropt (p = 1, (br, kr, bl, kl) = χ) transfer_from_left[br, kr] :=
        X[bl, kl] * A[kl, kr, p] * conj(B)[bl, br, p]
end
function transfer_from_left(
    X::UniformScaling{T},
    A::AbstractTensor3,
    B::AbstractTensor3 = A,
) where {T}
    @tensoropt (p = 1, (br, kr, l) = χ) transfer_from_left[br, kr] :=
        one(T) * A[l, kr, p] * conj(B)[l, br, p]
end
function transfer_from_left(
    X::AbstractMatrix,
    O::AbstractMatrix,
    A::AbstractTensor3,
    B::AbstractTensor3 = A,
)
    @tensoropt ((bp, kp) = 1, (br, kr, bl, kl) = χ) transfer_from_left[br, kr] :=
        X[bl, kl] * A[kl, kr, kp] * conj(B)[bl, br, bp] * O[bp, kp]
end
function transfer_from_left(
    X::UniformScaling{T},
    O::AbstractMatrix,
    A::AbstractTensor3,
    B::AbstractTensor3 = A,
) where {T}
    @tensoropt ((bp, kp) = 1, (br, kr, l) = χ) transfer_from_left[br, kr] :=
        one(T) * A[l, kr, kp] * conj(B)[l, br, bp] * O[bp, kp]
end

"""
    transfer_from_left(X, [O,] (AL, AR), [(BL, BR)=(AL, AR)])

```
   Λ
  ╱ ╲  ┌────────┐  ┌────────┐
┌┤ X ├─┤   AL   ├──┤   AR   ├──
│ ╲ ╱  └────┬───┘  └────┬───┘
│  V        │           │
│          .┴───────────┴.
│         (       O       )
│          `┬───────────┬'
│           │           │
│      ┌────┴───┐  ┌────┴───┐
└──────┤conj(BL)├──┤conj(BR)├─┐
       └────────┘  └────────┘ │
──────────────────────────────┘
```
"""
function transfer_from_left(
    X::AbstractMatrix,
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
)
    (AL, AR), (BL, BR) = A, B
    @tensoropt ((lp, rp) = 1, (br, kr, bl, kl, b, k) = χ) transfer_from_left[br, kr] :=
        X[bl, kl] *
        AL[kl, k, lp] *
        AR[k, kr, rp] *
        conj(BL)[bl, b, lp] *
        conj(BR)[b, br, rp]
end
function transfer_from_left(
    X::UniformScaling{T},
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
) where {T}
    (AL, AR), (BL, BR) = A, B
    @tensoropt ((lp, rp) = 1, (br, kr, l, b, k) = χ) transfer_from_left[br, kr] :=
        one(T) * AL[l, k, lp] * AR[k, kr, rp] * conj(BL)[l, b, lp] * conj(BR)[b, br, rp]
end
function transfer_from_left(
    X::AbstractMatrix,
    O::AbstractTensor4,
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
)
    (AL, AR), (BL, BR) = A, B
    @tensoropt ((blp, brp, klp, krp) = 1, (br, kr, bl, kl, b, k) = χ) transfer_from_left[
        br,
        kr,
    ] :=
        X[bl, kl] *
        AL[kl, k, klp] *
        AR[k, kr, krp] *
        conj(BL)[bl, b, blp] *
        conj(BR)[b, br, brp] *
        O[blp, brp, klp, krp]
end
function transfer_from_left(
    X::UniformScaling{T},
    O::AbstractTensor4,
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
) where {T}
    (AL, AR), (BL, BR) = A, B
    @tensoropt ((blp, brp, klp, krp) = 1, (br, kr, l, b, k) = χ) transfer_from_left[
        br,
        kr,
    ] :=
        one(T) *
        AL[l, k, klp] *
        AR[k, kr, krp] *
        conj(BL)[l, b, blp] *
        conj(BR)[b, br, brp] *
        O[blp, brp, klp, krp]
end


"""
    transfer_from_right(X, [O,] A, [B=A])

```
              Λ
  ┌───────┐  ╱ ╲
──┤   A   ├─┤ X ├─┐
  └───┬───┘  ╲ ╱  │
      │       V   │
     .┴.          │
    ( O )         │
     `┬'          │
      │           │
  ┌───┴───┐       │
┌─┤conj(B)├───────┘
│ └───────┘
└──────────────────
```
"""
function transfer_from_right(X::AbstractMatrix, A::AbstractTensor3, B::AbstractTensor3 = A)
    @tensoropt (p = 1, (br, kr, bl, kl) = χ) transfer_from_right[kl, bl] :=
        A[kl, kr, p] * conj(B)[bl, br, p] * X[kr, br]
end
function transfer_from_right(
    X::UniformScaling{T},
    A::AbstractTensor3,
    B::AbstractTensor3 = A,
) where {T}
    @tensoropt (p = 1, (r, bl, kl) = χ) transfer_from_right[kl, bl] :=
        one(T) * A[kl, r, p] * conj(B)[bl, r, p]
end
function transfer_from_right(
    X::AbstractMatrix,
    O::AbstractMatrix,
    A::AbstractTensor3,
    B::AbstractTensor3 = A,
)
    @tensoropt ((bp, kp) = 1, (br, kr, bl, kl) = χ) transfer_from_right[kl, bl] :=
        A[kl, kr, kp] * conj(B)[bl, br, bp] * X[kr, br] * O[bp, kp]
end
function transfer_from_right(
    X::UniformScaling{T},
    O::AbstractMatrix,
    A::AbstractTensor3,
    B::AbstractTensor3 = A,
) where {T}
    @tensoropt ((bp, kp) = 1, (r, bl, kl) = χ) transfer_from_right[kl, bl] :=
        one(T) * A[kl, r, kp] * conj(B)[bl, r, bp] * O[bp, kp]
end

"""
    transfer_from_right(X, [O,] (AL, AR), [(BL, BR)=(AL, AR)])

```
                           Λ
  ┌────────┐  ┌────────┐  ╱ ╲
──┤   AL   ├──┤   AR   ├─┤ X ├┐
  └────┬───┘  └────┬───┘  ╲ ╱ │
       │           │       V  │
      .┴───────────┴.         │
     (       O       )        │
      `┬───────────┬'         │
       │           │          │
  ┌────┴───┐  ┌────┴───┐      │
┌─┤conj(BL)├──┤conj(BR)├──────┘
│ └────────┘  └────────┘
└──────────────────────────────
```
"""
function transfer_from_right(
    X::AbstractMatrix,
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
)
    (AL, AR), (BL, BR) = A, B
    @tensoropt ((lp, rp) = 1, (br, kr, bl, kl, b, k) = χ) transfer_from_right[kl, bl] :=
        AL[kl, k, lp] *
        AR[k, kr, rp] *
        conj(BL)[bl, b, lp] *
        conj(BR)[b, br, rp] *
        X[kr, br]
end
function transfer_from_right(
    X::UniformScaling{T},
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
) where {T}
    (AL, AR), (BL, BR) = A, B
    @tensoropt ((lp, rp) = 1, (r, bl, kl, b, k) = χ) transfer_from_right[kl, bl] :=
        one(T) * AL[kl, k, lp] * AR[k, r, rp] * conj(BL)[bl, b, lp] * conj(BR)[b, r, rp]
end
function transfer_from_right(
    X::AbstractMatrix,
    O::AbstractTensor4,
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
)
    (AL, AR), (BL, BR) = A, B
    @tensoropt ((blp, brp, klp, krp) = 1, (br, kr, bl, kl, b, k) = χ) transfer_from_right[
        kl,
        bl,
    ] :=
        AL[kl, k, klp] *
        AR[k, kr, krp] *
        conj(BL)[bl, b, blp] *
        conj(BR)[b, br, brp] *
        X[kr, br] *
        O[blp, brp, klp, krp]
end
function transfer_from_right(
    X::UniformScaling{T},
    O::AbstractTensor4,
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
) where {T}

    (AL, AR), (BL, BR) = A, B
    @tensoropt ((blp, brp, klp, krp) = 1, (r, bl, kl, b, k) = χ) transfer_from_right[
        kl,
        bl,
    ] :=
        one(T) *
        AL[kl, k, klp] *
        AR[k, r, krp] *
        conj(BL)[bl, b, blp] *
        conj(BR)[b, r, brp] *
        O[blp, brp, klp, krp]
end
