"""
    transfer_from_left(X, [O,] A, [B=A])

```
 -X--A----
|    O
 -conj(B)-
          |
----------
```
"""
function transfer_from_left(X::AbstractMatrix, A::AbstractTensor3, B::AbstractTensor3 = A)

    let (_, _br, _kr, _bl, _kl, _p) = ntuple(x -> nothing, 6)
        @tensoropt !(_p) _[_br, _kr] :=
            X[_bl, _kl] * A[_kl, _kr, _p] * conj(B)[_bl, _br, _p]
    end
end
function transfer_from_left(
    X::UniformScaling{T},
    A::AbstractTensor3,
    B::AbstractTensor3 = A,
) where {T}

    one(T) .* let (_, _br, _kr, _l, _p) = ntuple(x -> nothing, 5)
        @tensoropt !(_p) _[_br, _kr] := A[_l, _kr, _p] * conj(B)[_l, _br, _p]
    end
end
function transfer_from_left(
    X::AbstractMatrix,
    O::AbstractMatrix,
    A::AbstractTensor3,
    B::AbstractTensor3 = A,
)

    let (_, _br, _kr, _bl, _kl, _bp, _kp) = ntuple(x -> nothing, 7)
        @tensoropt !(_bp, _kp) _[_br, _kr] :=
            X[_bl, _kl] * A[_kl, _kr, _kp] * conj(B)[_bl, _br, _bp] * O[_bp, _kp]
    end
end
function transfer_from_left(
    X::UniformScaling{T},
    O::AbstractMatrix,
    A::AbstractTensor3,
    B::AbstractTensor3 = A,
) where {T}

    one(T) .* let (_, _br, _kr, _l, _bp, _kp) = ntuple(x -> nothing, 6)
        @tensoropt !(_bp, _kp) _[_br, _kr] :=
            A[_l, _kr, _kp] * conj(B)[_l, _br, _bp] * O[_bp, _kp]
    end
end

"""
    transfer_from_left(X, [O,] (AL, AR), [(BL, BR)=(AL, AR)])

```
 -X--AL-------AR----
|    [OOOOOOOOO]
 -conj(BL)-conj(BR)-
                    |
--------------------
```
"""
function transfer_from_left(
    X::AbstractMatrix,
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
)

    (AL, AR), (BL, BR) = A, B
    let (_, _br, _kr, _bl, _kl, _b, _k, _lp, _rp) = ntuple(x -> nothing, 9)
        @tensoropt !(_lp, _rp) _[_br, _kr] :=
            X[_bl, _kl] *
            AL[_kl, _k, _lp] *
            AR[_k, _kr, _rp] *
            conj(BL)[_bl, _b, _lp] *
            conj(BR)[_b, _br, _rp]
    end
end
function transfer_from_left(
    X::UniformScaling{T},
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
) where {T}

    (AL, AR), (BL, BR) = A, B
    one(T) .* let (_, _br, _kr, _l, _b, _k, _lp, _rp) = ntuple(x -> nothing, 8)
        @tensoropt !(_lp, _rp) _[_br, _kr] :=
            AL[_l, _k, _lp] *
            AR[_k, _kr, _rp] *
            conj(BL)[_l, _b, _lp] *
            conj(BR)[_b, _br, _rp]
    end
end
function transfer_from_left(
    X::AbstractMatrix,
    O::AbstractTensor4,
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
)

    (AL, AR), (BL, BR) = A, B
    let (_, _br, _kr, _bl, _kl, _b, _k, _blp, _brp, _klp, _krp) = ntuple(x -> nothing, 11)
        @tensoropt !(_blp, _brp, _klp, _krp) _[_br, _kr] :=
            X[_bl, _kl] *
            AL[_kl, _k, _klp] *
            AR[_k, _kr, _krp] *
            conj(BL)[_bl, _b, _blp] *
            conj(BR)[_b, _br, _brp] *
            O[_blp, _brp, _klp, _krp]
    end
end
function transfer_from_left(
    X::UniformScaling{T},
    O::AbstractTensor4,
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
) where {T}

    (AL, AR), (BL, BR) = A, B
    one(T) .*
    let (_, _br, _kr, _l, _b, _k, _blp, _brp, _klp, _krp) = ntuple(x -> nothing, 10)
        @tensoropt !(_blp, _brp, _klp, _krp) _[_br, _kr] :=
            AL[_l, _k, _klp] *
            AR[_k, _kr, _krp] *
            conj(BL)[_l, _b, _blp] *
            conj(BR)[_b, _br, _brp] *
            O[_blp, _brp, _klp, _krp]
    end
end


"""
    transfer_from_right(X, [O,] A, [B=A])

```
-----A--X-
     O    |
 -conj(B)-
|
 ----------
```
"""
function transfer_from_right(X::AbstractMatrix, A::AbstractTensor3, B::AbstractTensor3 = A)

    let (_, _br, _kr, _bl, _kl, _p) = ntuple(x -> nothing, 6)
        @tensoropt !(_p) _[_kl, _bl] :=
            A[_kl, _kr, _p] * conj(B)[_bl, _br, _p] * X[_kr, _br]
    end
end
function transfer_from_right(
    X::UniformScaling{T},
    A::AbstractTensor3,
    B::AbstractTensor3 = A,
) where {T}

    one(T) .* let (_, _r, _bl, _kl, _p) = ntuple(x -> nothing, 5)
        @tensoropt !(_p) _[_kl, _bl] := A[_kl, _r, _p] * conj(B)[_bl, _r, _p]
    end
end
function transfer_from_right(
    X::AbstractMatrix,
    O::AbstractMatrix,
    A::AbstractTensor3,
    B::AbstractTensor3 = A,
)

    let (_, _br, _kr, _bl, _kl, _bp, _kp) = ntuple(x -> nothing, 7)
        @tensoropt !(_bp, _kp) _[_kl, _bl] :=
            A[_kl, _kr, _kp] * conj(B)[_bl, _br, _bp] * X[_kr, _br] * O[_bp, _kp]
    end
end
function transfer_from_right(
    X::UniformScaling{T},
    O::AbstractMatrix,
    A::AbstractTensor3,
    B::AbstractTensor3 = A,
) where {T}

    one(T) .* let (_, _r, _bl, _kl, _bp, _kp) = ntuple(x -> nothing, 6)
        @tensoropt !(_bp, _kp) _[_kl, _bl] :=
            A[_kl, _r, _kp] * conj(B)[_bl, _r, _bp] * O[_bp, _kp]
    end
end

"""
    transfer_from_right(X, [O,] (AL, AR), [(BL, BR)=(AL, AR)])

```
-----AL-------AR--X-
     [OOOOOOOOO]    |
 -conj(BL)-conj(BR)-
|
 --------------------
```
"""
function transfer_from_right(
    X::AbstractMatrix,
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
)

    (AL, AR), (BL, BR) = A, B
    let (_, _br, _kr, _bl, _kl, _b, _k, _lp, _rp) = ntuple(x -> nothing, 9)
        @tensoropt !(_lp, _rp) _[_kl, _bl] :=
            AL[_kl, _k, _lp] *
            AR[_k, _kr, _rp] *
            conj(BL)[_bl, _b, _lp] *
            conj(BR)[_b, _br, _rp] *
            X[_kr, _br]
    end
end
function transfer_from_right(
    X::UniformScaling{T},
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
) where {T}

    (AL, AR), (BL, BR) = A, B
    one(T) .* let (_, _r, _bl, _kl, _b, _k, _lp, _rp) = ntuple(x -> nothing, 8)
        @tensoropt !(_lp, _rp) _[_kl, _bl] :=
            AL[_kl, _k, _lp] *
            AR[_k, _r, _rp] *
            conj(BL)[_bl, _b, _lp] *
            conj(BR)[_b, _r, _rp]
    end
end
function transfer_from_right(
    X::AbstractMatrix,
    O::AbstractTensor4,
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
)

    (AL, AR), (BL, BR) = A, B
    let (_, _br, _kr, _bl, _kl, _b, _k, _blp, _brp, _klp, _krp) = ntuple(x -> nothing, 11)
        @tensoropt !(_blp, _brp, _klp, _krp) _[_kl, _bl] :=
            AL[_kl, _k, _klp] *
            AR[_k, _kr, _krp] *
            conj(BL)[_bl, _b, _blp] *
            conj(BR)[_b, _br, _brp] *
            X[_kr, _br] *
            O[_blp, _brp, _klp, _krp]
    end
end
function transfer_from_right(
    X::UniformScaling{T},
    O::AbstractTensor4,
    A::Tuple{<:AbstractTensor3,<:AbstractTensor3},
    B::Tuple{<:AbstractTensor3,<:AbstractTensor3} = A,
) where {T}

    (AL, AR), (BL, BR) = A, B
    one(T) .*
    let (_, _r, _bl, _kl, _b, _k, _blp, _brp, _klp, _krp) = ntuple(x -> nothing, 10)
        @tensoropt !(_blp, _brp, _klp, _krp) _[_kl, _bl] :=
            AL[_kl, _k, _klp] *
            AR[_k, _r, _krp] *
            conj(BL)[_bl, _b, _blp] *
            conj(BR)[_b, _r, _brp] *
            O[_blp, _brp, _klp, _krp]
    end
end
