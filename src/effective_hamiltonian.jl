"""
    mul_matrix_from_left(A, X)

```
-X-A-
   |
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
-A-X-
 |
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
-A-
 O
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
 ---AL------A-
|   [OOOOOOO] |
 conj(AL)-  | |
          | | |
 ---------  |  -
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
  -A------AR----
 | [OOOOOOO]    |
 | |  -conj(AR)-
 | | |
-  |  -----------
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
