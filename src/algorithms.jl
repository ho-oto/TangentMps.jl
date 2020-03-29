"""
    ibcleft(O, AL, C; kwargs...)

Compute left infinite boundary condition, defined by
```
   ┌────────┐   ┌────────┐      ┌─            ┌────────┐    ─┐ P
┌──┤   AL   ├───┤   AL   ├───  ┌┘          ───┤   AL   ├───  └┐  ───
│  └────┬───┘   └────┬───┘     │   ┌┐         └────┬───┘      │
│       .────────────┴.        │  ─┘│              │          │
│      (       O       )       │    │  ───         │          │
│       `────────────┬'        │    │              │          │
│  ┌────┴───┐   ┌────┴───┐     │  ──┴──       ┌────┴───┐      │
└──┤conj(AL)├───┤conj(AL)├───  └┐          ───┤conj(AL)├───  ┌┘  ───
   └────────┘   └────────┘      └─            └────────┘    ─┘
```

### Return values:
`ibc, info = ibcleft(O, AL, C)`

### Arguments:
*   `O`: Hamiltonian (only two-site operator is supported)
*   `AL`: left-canonical tensor of mixed-canonical uniform MPS representation
*   `C`: center matrix of mixed-canonical uniform MPS representation

If `O`, `AL` and `C` is `AbstractArray` oblect, nothing have to be done. The following
methods should be defined:

*   `TangentMps.transfer_from_left(X, O, AL)`
*   `TangentMps.transfer_from_left(X, O, (AL, AL))`
*   `TangentMps.similar_normalized_eye(X)`
*   `Base.*(X, Y)`
*   `Base.*(x::Number, X)`
*   `Base.+(X, Y)`
*   `Base.adjoint(X)`
*   `Base.eltype(X)`
*   `Base.similar(X, [T::Type<:Number])`
*   `Base.fill!(X, α::Number)`
*   `Base.copyto!(X, Y)`
*   `LinearAlgebra.mul!(X, Y, α::Number)`
*   `LinearAlgebra.rmul!(X, α::Number)`
*   `LinearAlgebra.axpy!(α::Number, X, Y)`
*   `LinearAlgebra.axpby!(α::Number, X, β::Number, Y)`
*   `LinearAlgebra.tr(X)`
*   `LinearAlgebra.dot(X,Y)`
*   `LinearAlgebra.norm(X)`

where `typeof(X) == typeof(Y) == typeof(C)` is satisfied.

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
function ibcleft(
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
        Y - transfer_from_left(Y, AL) + tr(C' * Y * C) * eye
    end
    inf.converged == 1 || @warn "L un-converged"
    ibc, inf
end

"""
    ibcright(O, AR, C; kwargs...)

Compute left infinite boundary condition, defined by
```
     ┌─            ┌────────┐    ─┐ P    ┌────────┐   ┌────────┐
─── ┌┘          ───┤   AR   ├───  └┐  ───┤   AR   ├───┤   AR   ├──┐
    │   ┌┐         └────┬───┘      │     └────┬───┘   └────┬───┘  │
    │  ─┘│              │          │         .┴────────────.      │
    │    │  ───         │          │        (       O       )     │
    │    │              │          │         `┬────────────'      │
    │  ──┴──       ┌────┴───┐      │     ┌────┴───┐   ┌────┴───┐  │
─── └┐          ───┤conj(AR)├───  ┌┘  ───┤conj(AR)├───┤conj(AR)├──┘
     └─            └────────┘    ─┘      └────────┘   └────────┘
```

### Return values:
`ibc, info = ibcright(O, AR, C)`

### Arguments:
*   `O`: Hamiltonian (only two-site operator is supported)
*   `AR`: right-canonical tensor of mixed-canonical uniform MPS representation
*   `C`: center matrix of mixed-canonical uniform MPS representation

If `O`, `AR` and `C` is `AbstractArray` oblect, nothing have to be done. The following
methods should be defined:
*   `TangentMps.transfer_from_right(X, O, AR)`
*   `TangentMps.transfer_from_right(X, O, (AR, AR))`
*   `TangentMps.similar_normalized_eye(X)`
*   `Base.*(X, Y)`
*   `Base.*(x::Number, X)`
*   `Base.+(X, Y)`
*   `Base.adjoint(X)`
*   `Base.eltype(X)`
*   `Base.similar(X, [T::Type<:Number])`
*   `Base.fill!(X, α::Number)`
*   `Base.copyto!(X, Y)`
*   `LinearAlgebra.mul!(X, Y, α::Number)`
*   `LinearAlgebra.rmul!(X, α::Number)`
*   `LinearAlgebra.axpy!(α::Number, X, Y)`
*   `LinearAlgebra.axpby!(α::Number, X, β::Number, Y)`
*   `LinearAlgebra.tr(X)`
*   `LinearAlgebra.dot(X,Y)`
*   `LinearAlgebra.norm(X)`

where `typeof(X) == typeof(Y) == typeof(C)` is satisfied.

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
function ibcright(
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
        Y - transfer_from_right(Y, AR) + tr(C * Y * C') * eye
    end
    inf.converged == 1 || @warn "R un-converged"
    ibc, inf
end

"""
    vumpsstep(O, AL, AR, AC, C; kwargs...)
    vumpsstep(O, AL, AC, C; kwargs...) # with inversion symmetry

### Return values:
    `(AL, AR, AC, C), (ϵL, ϵR), info = vumpsstep(O, AL, AR, AC, C; kwargs...)`
    `(AL, AC, C), (ϵL,), info = vumpsstep(O, AL, AC, C; kwargs...)`
"""
function vumpsstep(
    O,
    AL,
    AR,
    AC,
    C;
    ibcatol = KrylovDefaults.tol,
    ibcrtol = KrylovDefaults.tol,
    ibckrylovdim = KrylovDefaults.krylovdim,
    ibcmaxiter = KrylovDefaults.maxiter,
    eigtol = KrylovDefaults.tol,
    eigkrylovdim = KrylovDefaults.krylovdim,
    eigmaxiter = KrylovDefaults.maxiter,
)

    (L, infL) = ibcleft(
        O,
        AL,
        C;
        atol = ibcatol,
        rtol = ibcrtol,
        krylovdim = ibckrylovdim,
        maxiter = ibcmaxiter,
    )
    (R, infR) = ibcright(
        O,
        AR,
        C;
        atol = ibcatol,
        rtol = ibcrtol,
        krylovdim = ibckrylovdim,
        maxiter = ibcmaxiter,
    )

    (valAC,), (vecAC,), infAC = eigsolve(
        AC,
        1,
        :SR;
        ishermitian = true,
        tol = eigtol,
        krylovdim = eigkrylovdim,
        maxiter = eigmaxiter,
    ) do X
        mul_matrix_from_left(X, L) +
        mul_matrix_from_right(X, R) +
        mul_operator_with_left(X, O, AL) +
        mul_operator_with_right(X, O, AR)
    end
    infAC.converged ≥ 1 || @warn "AC un-converged"
    norm(vecAC) ≈ 1 && (vecAC /= norm(vecAC))

    (valC,), (vecC,), infC = eigsolve(
        C,
        1,
        :SR;
        ishermitian = true,
        tol = eigtol,
        krylovdim = eigkrylovdim,
        maxiter = eigmaxiter,
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
    infC.converged ≥ 1 || @warn "C un-converged"
    norm(vecC) ≈ 1 && (vecC /= norm(vecC))

    AL_, ϵL = al_from_ac_and_c(vecAC, vecC)
    AR_, ϵR = ar_from_ac_and_c(vecAC, vecC)

    (AL_, AR_, vecAC, vecC), (ϵL, ϵR), (infL, infR, infAC, infC)
end
function vumpsstep(
    O,
    AL,
    AC,
    C;
    ibc_atol = KrylovDefaults.tol,
    ibc_rtol = KrylovDefaults.tol,
    ibc_krylovdim = KrylovDefaults.krylovdim,
    ibc_maxiter = KrylovDefaults.maxiter,
    eig_tol = KrylovDefaults.tol,
    eig_krylovdim = KrylovDefaults.krylovdim,
    eig_maxiter = KrylovDefaults.maxiter,
)

    AR = permutedims(AL, (2, 1, 3))

    (L, infL) = ibcleft(
        O,
        AL,
        C;
        atol = ibc_atol,
        rtol = ibc_rtol,
        krylovdim = ibc_krylovdim,
        maxiter = ibc_maxiter,
    )
    R = transpose(L)

    (valAC,), (vecAC,), infAC = eigsolve(
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
        mul_operator_with_right(X, O, AR)
    end
    infAC.converged ≥ 1 || @warn "AC un-converged"
    norm(vecAC) ≈ 1 && (vecAC /= norm(vecAC))

    (valC,), (vecC,), infC = eigsolve(
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
        )
    end
    infC.converged ≥ 1 || @warn "C un-converged"
    norm(vecC) ≈ 1 && (vecC /= norm(vecC))

    AL_, ϵL = al_from_ac_and_c(vecAC, vecC)

    (AL_, vecAC, vecC), (ϵL,), (infL, infAC, infC)
end

"""
    tdvpstep(O, dt, AL, AR, AC, C; kwargs...)
    tdvpstep(O, dt, AL, AC, C; kwargs...) # with inversion symmetry

### Return values:
    `(AL, AR, AC, C), norm, (ϵL, ϵR), info = tdvpstep(O, AL, AR, AC, C; kwargs...)`
    `(AL, AC, C), norm, (ϵL,), info = tdvpstep(O, AL, AC, C; kwargs...)`
where `norm` denotes `|| exp(dt*O)|Ψ(AL, AR, AC, C)⟩ || / || |Ψ(AL, AR, AC, C)⟩ ||`
"""
function tdvpstep(
    O,
    dt,
    AL,
    AR,
    AC,
    C;
    ishermitian,
    ibc_atol = KrylovDefaults.tol,
    ibc_rtol = KrylovDefaults.tol,
    ibc_krylovdim = KrylovDefaults.krylovdim,
    ibc_maxiter = KrylovDefaults.maxiter,
    eig_tol = KrylovDefaults.tol,
    eig_krylovdim = KrylovDefaults.krylovdim,
    eig_maxiter = KrylovDefaults.maxiter,
)
    EO = tr(transfer_from_left(I, O, (AL, AC)))
    O_ = reshape(reshape(O, size(O, 1) * size(O, 2), :) - EO * I, size(O))

    (L, infL) = ibcleft(
        O,
        AL,
        C;
        atol = ibc_atol,
        rtol = ibc_rtol,
        krylovdim = ibc_krylovdim,
        maxiter = ibc_maxiter,
    )
    (R, infR) = ibcright(
        O,
        AR,
        C;
        atol = ibc_atol,
        rtol = ibc_rtol,
        krylovdim = ibc_krylovdim,
        maxiter = ibc_maxiter,
    )
    L = L - tr(transfer_from_left(L, AC)) * I
    R = R - tr(transfer_from_right(R, AC)) * I

    vecAC, infAC = exponentiate(
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
    infAC.converged == 1 || @warn "AC un-converged"
    normAC = norm(vecAC)
    vecAC /= normAC
    normAC *= exp(dt * EO)

    vecC, infC = exponentiate(
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
    infC.converged == 1 || @warn "C un-converged"
    normC = norm(vecC)
    vecC /= normC
    normC *= exp(dt * EO)

    AL_, ϵL = al_from_ac_and_c(vecAC, vecC)
    AR_, ϵR = ar_from_ac_and_c(vecAC, vecC)

    (al = AL_, ar = AR_, ac = vecAC, c = vecC),
    (ac = normAC, c = normC),
    (l = ϵL, r = ϵR),
    (l = infL, r = infR, ac = infAC, c = infC)
end
function tdvpstep(
    O,
    dt,
    AL,
    AC,
    C;
    ishermitian,
    ibc_atol = KrylovDefaults.tol,
    ibc_rtol = KrylovDefaults.tol,
    ibc_krylovdim = KrylovDefaults.krylovdim,
    ibc_maxiter = KrylovDefaults.maxiter,
    eig_tol = KrylovDefaults.tol,
    eig_krylovdim = KrylovDefaults.krylovdim,
    eig_maxiter = KrylovDefaults.maxiter,
)
    AR = permutedims(AL, (2, 1, 3))
    EO = tr(transfer_from_left(I, O, (AL, AC)))
    O_ = reshape(reshape(O, size(O, 1) * size(O, 2), :) - EO * I, size(O))

    (L, infL) = ibcleft(
        O_,
        AL,
        C;
        atol = ibc_atol,
        rtol = ibc_rtol,
        krylovdim = ibc_krylovdim,
        maxiter = ibc_maxiter,
    )
    L = L - tr(transfer_from_left(L, AC)) * I
    R = transpose(L)

    vecAC, infAC = exponentiate(
        dt,
        AC;
        ishermitian = ishermitian,
        tol = eig_tol,
        krylovdim = eig_krylovdim,
        maxiter = eig_maxiter,
    ) do X
        mul_matrix_from_left(X, L) +
        mul_matrix_from_right(X, R) +
        mul_operator_with_left(X, O_, AL) +
        mul_operator_with_right(X, O_, AR)
    end
    infAC.converged == 1 || @warn "AC un-converged"
    normAC = norm(vecAC)
    vecAC /= normAC
    normAC *= exp(dt * EO)

    vecC, infC = exponentiate(
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
            mul_operator_with_left(Y, O_, AL) +
            mul_operator_with_right(Y, O_, AR),
            AR,
        )
    end
    infC.converged == 1 || @warn "C un-converged"
    normC = norm(vecC)
    vecC /= normC
    normC *= exp(dt * EO)

    AL_, ϵL = al_from_ac_and_c(vecAC, vecC)

    (al = AL_, ac = vecAC, c = vecC),
    (ac = normAC, c = normC),
    (l = ϵL,),
    (l = infL, ac = infAC, c = infC)
end
