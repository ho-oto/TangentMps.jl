"""
    ibc_left(O, AL, C; kwargs...)

Compute left infinite boundary condition, defined by
```
   ┌────────┐   ┌────────┐      ┌─            ┌────────┐    ─┐ P
┌──│   AL   │───│   AL   │───  ┌┘          ───│   AL   │───  └┐  ───
│  └────────┘   └────────┘     │   ┌┐         └────────┘      │
│       │            │         │  ─┘│              │          │
│     ┌─┴────────────┴┐        │    │              │          │
│     │       O       │        │    │  ───         │          │
│     └─┬────────────┬┘        │    │              │          │
│       │            │         │    │              │          │
│  ┌────────┐   ┌────────┐     │  ──┴──       ┌────────┐      │
└──│conj(AL)│───│conj(AL)│───  └┐          ───│conj(AL)│───  ┌┘  ───
   └────────┘   └────────┘      └─            └────────┘    ─┘
```

### Return values:
`ibc, info = ibc_left(O, AL, C)`

### Arguments:
*   `O`: Hamiltonian (only two-site operator is supported)
*   `AL`: left-canonical tensor of mixed-canonical uniform MPS representation
*   `C`: center matrix of mixed-canonical uniform MPS representation

If `O`, `AL` and `C` is `AbstractArray` oblect, nothing have to be done.
`TangentMps.transfer_from_left(X, O, AL)`,
`TangentMps.transfer_from_left(X, O, (AL, AL))`,
`TangentMps.similar_normalized_eye(X)`,
`Base.*(X, Y)`,
`Base.*(x::Number, X)`,
`Base.+(X, Y)`,
`Base.adjoint(X)`,
`Base.eltype(X)`,
`Base.similar(X, [T::Type<:Number])`,
`Base.fill!(X, α::Number)`,
`Base.copyto!(X, Y)`,
`LinearAlgebra.mul!(X, Y, α::Number)`,
`LinearAlgebra.rmul!(X, α::Number)`,
`LinearAlgebra.axpy!(α::Number, X, Y)`,
`LinearAlgebra.axpby!(α::Number, X, β::Number, Y)`,
`LinearAlgebra.tr(X)`,
`LinearAlgebra.dot(X,Y)` and
`LinearAlgebra.norm(X)`
should be defined where `typeof(X) == typeof(Y) == typeof(C)` is satisfied.

### Keyword arguments:
Keyword arguments are passed to `KrylovKit.linsolve` used in `ibc_left`.
*   `atol::Real`: the requested accuracy, i.e. absolute tolerance, on the norm of the
    residual.
*   `rtol::Real`: the requested accuracy on the norm of the residual, relative to the norm
    of the right hand side `b`.
*   `tol::Real`: the requested accuracy on the norm of the residual which is actually used,
    but which defaults to `max(atol, rtol*norm(b))`. So either `atol` and `rtol` or directly
    use `tol` (in which case the value of `atol` and `rtol` will be ignored).
*   `krylovdim::Integer`: the maximum dimension of the Krylov subspace that will be
    constructed.
*   `maxiter::Integer`: the number of times the Krylov subspace can be rebuilt; see below for
    further details on the algorithms.
"""
function ibc_left(
    O,
    AL,
    C;
    atol = KrylovDefaults.tol,
    rtol = KrylovDefaults.tol,
    krylovdim = KrylovDefaults.krylovdim,
    maxiter = KrylovDefaults.maxiter,
)
    L = transfer_from_left(I, O, (AL, AL))
    eye = similar_normalized_eye(C)
    rhs = L - tr(C' * L * C) * eye
    ibc, inf = linsolve(
        rhs;
        atol = atol,
        rtol = rtol,
        krylovdim = krylovdim,
        maxiter = maxiter,
    ) do Y
        Y - transfer_from_left(Y, AL) + tr(C' * Y * C) * eye #TODO: replace by in-place operation?
    end
    inf.converged == 1 || @warn "L un-converged"
    ibc, inf
end

"""
    ibc_right(O, AR, C; kwargs...)

Compute left infinite boundary condition, defined by
```
     ┌─            ┌────────┐    ─┐ P    ┌────────┐   ┌────────┐
─── ┌┘          ───│   AR   │───  └┐  ───│   AR   │───│   AR   │──┐
    │   ┌┐         └────────┘      │     └────────┘   └────────┘  │
    │  ─┘│              │          │          │            │      │
    │    │              │          │        ┌─┴────────────┴┐     │
    │    │  ───         │          │        │       O       │     │
    │    │              │          │        └─┬────────────┬┘     │
    │    │              │          │          │            │      │
    │  ──┴──       ┌────────┐      │     ┌────────┐   ┌────────┐  │
─── └┐          ───│conj(AR)│───  ┌┘  ───│conj(AR)│───│conj(AR)│──┘
     └─            └────────┘    ─┘      └────────┘   └────────┘
```

### Return values:
`ibc, info = ibc_right(O, AR, C)`

### Arguments:
*   `O`: Hamiltonian (only two-site operator is supported)
*   `AR`: right-canonical tensor of mixed-canonical uniform MPS representation
*   `C`: center matrix of mixed-canonical uniform MPS representation

If `O`, `AR` and `C` is `AbstractArray` oblect, nothing have to be done.
`TangentMps.transfer_from_right(X, O, AR)`,
`TangentMps.transfer_from_right(X, O, (AR, AR))`,
`TangentMps.similar_normalized_eye(X)`,
`Base.*(X, Y)`,
`Base.*(x::Number, X)`,
`Base.+(X, Y)`,
`Base.adjoint(X)`,
`Base.eltype(X)`,
`Base.similar(X, [T::Type<:Number])`,
`Base.fill!(X, α::Number)`,
`Base.copyto!(X, Y)`,
`LinearAlgebra.mul!(X, Y, α::Number)`,
`LinearAlgebra.rmul!(X, α::Number)`,
`LinearAlgebra.axpy!(α::Number, X, Y)`,
`LinearAlgebra.axpby!(α::Number, X, β::Number, Y)`,
`LinearAlgebra.tr(X)`,
`LinearAlgebra.dot(X,Y)` and
`LinearAlgebra.norm(X)`
should be defined where `typeof(X) == typeof(Y) == typeof(C)` is satisfied.

### Keyword arguments:
Keyword arguments are passed to `KrylovKit.linsolve` used in `ibc_left`.
*   `atol::Real`: the requested accuracy, i.e. absolute tolerance, on the norm of the
    residual.
*   `rtol::Real`: the requested accuracy on the norm of the residual, relative to the norm
    of the right hand side `b`.
*   `tol::Real`: the requested accuracy on the norm of the residual which is actually used,
    but which defaults to `max(atol, rtol*norm(b))`. So either `atol` and `rtol` or directly
    use `tol` (in which case the value of `atol` and `rtol` will be ignored).
*   `krylovdim::Integer`: the maximum dimension of the Krylov subspace that will be
    constructed.
*   `maxiter::Integer`: the number of times the Krylov subspace can be rebuilt; see below for
    further details on the algorithms.
"""
function ibc_right(
    O,
    AR,
    C;
    atol = KrylovDefaults.tol,
    rtol = KrylovDefaults.tol,
    krylovdim = KrylovDefaults.krylovdim,
    maxiter = KrylovDefaults.maxiter,
)
    R = transfer_from_right(I, O, (AR, AR))
    eye = similar_normalized_eye(C)
    rhs = R - tr(C * R * C') * eye
    ibc, inf = linsolve(
        rhs;
        atol = atol,
        rtol = rtol,
        krylovdim = krylovdim,
        maxiter = maxiter,
    ) do Y
        Y - transfer_from_right(Y, AR) + tr(C * Y * C') * eye #TODO: replace by in-place operation?
    end
    inf.converged == 1 || @warn "R un-converged"
    ibc, inf
end

"""
    vumps_step(O, AL, AR, AC, C; kwargs...)
"""
function vumps_step(
    O,
    AL,
    AR,
    AC,
    C;
    isinvsym::Bool = false,
    ibc_atol = KrylovDefaults.tol,
    ibc_rtol = KrylovDefaults.tol,
    ibc_krylovdim = KrylovDefaults.krylovdim,
    ibc_maxiter = KrylovDefaults.maxiter,
    eig_tol = KrylovDefaults.tol,
    eig_krylovdim = KrylovDefaults.krylovdim,
    eig_maxiter = KrylovDefaults.maxiter,
)

    (L, inf_L) = ibc_left(
        O,
        AL,
        C;
        atol = ibc_atol,
        rtol = ibc_rtol,
        krylovdim = ibc_krylovdim,
        maxiter = ibc_maxiter,
    )
    (R, inf_R) = isinvsym ? (transpose(L), inf_L) :
        ibc_right( #TODO: check transpose/conj
        O,
        AR,
        C;
        atol = ibc_atol,
        rtol = ibc_rtol,
        krylovdim = ibc_krylovdim,
        maxiter = ibc_maxiter,
    )

    (val_AC,), (vec_AC,), inf_AC = eigsolve(
        AC,
        1,
        :SR;
        ishermitian = true,
        tol = eig_tol,
        krylovdim = eig_krylovdim,
        maxiter = eig_maxiter,
    ) do X
        mul_matrix_from_left(X, L) +
        mul_matrix_from_right(X, R) +
        mul_operator_with_left(X, O, AL) +
        mul_operator_with_right(X, O, AR) #TODO: replace by in-place operation?
    end
    inf_AC.converged ≥ 1 || @warn "AC un-converged"
    norm(vec_AC) ≈ 1 && (vec_AC /= norm(vec_AC))

    (val_C,), (vec_C,), inf_C = eigsolve(
        C,
        1,
        :SR;
        ishermitian = true,
        tol = eig_tol,
        krylovdim = eig_krylovdim,
        maxiter = eig_maxiter,
    ) do X
        Y = mul_matrix_from_left(AR, X)
        transfer_from_right(
            I,
            mul_matrix_from_left(Y, L) +
            mul_matrix_from_right(Y, R) +
            mul_operator_with_left(Y, O, AL) +
            mul_operator_with_right(Y, O, AR),
            AR,
        ) #TODO: replace by in-place operation?
    end
    inf_C.converged ≥ 1 || @warn "C un-converged"
    norm(vec_C) ≈ 1 && (vec_C /= norm(vec_C))

    (AL_, AR_), (epl, epr) = al_and_ar(vec_AC, vec_C)

    (AL_, AR_, vec_AC, vec_C), (epl, epr), (inf_L, inf_R, inf_AC, inf_C)
end

"""
    tdvp_step(O, dt, AL, AR, AC, C; kwargs...)
"""
function tdvp_step(
    O,
    dt,
    AL,
    AR,
    AC,
    C;
    ishermitian,
    isinvsym::Bool = false,
    ibc_atol = KrylovDefaults.tol,
    ibc_rtol = KrylovDefaults.tol,
    ibc_krylovdim = KrylovDefaults.krylovdim,
    ibc_maxiter = KrylovDefaults.maxiter,
    eig_tol = KrylovDefaults.tol,
    eig_krylovdim = KrylovDefaults.krylovdim,
    eig_maxiter = KrylovDefaults.maxiter,
)
    (L, inf_L) = ibc_left(
        O,
        AL,
        C;
        atol = ibc_atol,
        rtol = ibc_rtol,
        krylovdim = ibc_krylovdim,
        maxiter = ibc_maxiter,
    )
    (R, inf_R) = isinvsym ? (transpose(L), inf_L) :
        ibc_right(
        O,
        AR,
        C;
        atol = ibc_atol,
        rtol = ibc_rtol,
        krylovdim = ibc_krylovdim,
        maxiter = ibc_maxiter,
    )

    vec_AC, inf_AC = exponentiate(
        dt,
        AC;
        ishermitian = ishermitian,
        tol = eig_tol,
        krylovdim = eig_krylovdim,
        maxiter = eig_maxiter,
    ) do X
        mul_matrix_from_left(X, L) +
        mul_matrix_from_right(X, R) +
        mul_operator_with_left(X, O, AL) +
        mul_operator_with_right(X, O, AR)
    end
    inf_AC.converged == 1 || @warn "AC un-converged"
    norm_AC = norm(vec_AC)
    vec_AC /= norm_AC

    vec_C, inf_C = exponentiate(
        dt,
        C;
        ishermitian = ishermitian,
        tol = eig_tol,
        krylovdim = eig_krylovdim,
        maxiter = eig_maxiter,
    ) do X
        Y = mul_matrix_from_left(AR, X)
        transfer_from_right(
            I,
            mul_matrix_from_left(Y, L) +
            mul_matrix_from_right(Y, R) +
            mul_operator_with_left(Y, O, AL) +
            mul_operator_with_right(Y, O, AR),
            AR,
        )
    end
    inf_C.converged == 1 || @warn "C un-converged"
    norm_C = norm(vec_C)
    vec_C /= norm_C

    (AL_, AR_), (epl, epr) = al_and_ar(vec_AC, vec_C)

    (AL_, AR_, vec_AC, vec_C), (norm_AC, norm_C), (epl, epr), (inf_L, inf_R, inf_AC, inf_C)
end
