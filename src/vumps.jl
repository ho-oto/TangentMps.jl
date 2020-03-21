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
        Y - transfer_from_left(Y, AL) + tr(C' * Y * C) * eye
    end
    inf.converged == 1 || @warn "L un-converged"
    ibc, inf
end

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
        Y - transfer_from_right(Y, AR) + tr(C * Y * C') * eye
    end
    inf.converged == 1 || @warn "R un-converged"
    ibc, inf
end

function vumps_step(
    O,
    AL,
    AR,
    AC,
    C;
    tol = KrylovDefaults.tol,
    krylovdim = KrylovDefaults.krylovdim,
    maxiter = KrylovDefaults.maxiter,
)

    (L, inf_L), (R, inf_R) = ibc_left(O, AL, C), ibc_right(O, AR, C)

    (val_AC,), (vec_AC,), inf_AC = eigsolve(
        AC,
        1,
        :SR;
        ishermitian = true,
        tol = tol,
        krylovdim = krylovdim,
        maxiter = maxiter,
    ) do X
        mul_matrix_from_left(X, L) +
        mul_matrix_from_right(X, R) +
        mul_operator_with_left(X, O, AL) +
        mul_operator_with_right(X, O, AR)
    end
    inf_AC.converged ≥ 1 || @warn "AC un-converged"
    norm(vec_AC) ≈ 1 && (vec_AC /= norm(vec_AC))

    (val_C,), (vec_C,), inf_C = eigsolve(
        C,
        1,
        :SR;
        ishermitian = true,
        tol = tol,
        krylovdim = krylovdim,
        maxiter = maxiter,
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
    inf_C.converged ≥ 1 || @warn "C un-converged"
    norm(vec_C) ≈ 1 && (vec_C /= norm(vec_C))

    (AL_, AR_), (epl, epr) = al_and_ar(vec_AC, vec_C)

    (AL_, AR_, vec_AC, vec_C), (epl, epr), (inf_L, inf_R, inf_AC, inf_C)
end

function tdvp_step(
    O,
    dt,
    AL,
    AR,
    AC,
    C;
    ishermitian = true,
    tol = KrylovDefaults.tol,
    krylovdim = KrylovDefaults.krylovdim,
    maxiter = KrylovDefaults.maxiter,
)
    (L, inf_L), (R, inf_R) = ibc_left(O, AL, C), ibc_right(O, AR, C)

    vec_AC, inf_AC = exponentiate(
        dt,
        AC;
        ishermitian = ishermitian,
        tol = tol,
        krylovdim = krylovdim,
        maxiter = maxiter,
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
        tol = tol,
        krylovdim = krylovdim,
        maxiter = maxiter,
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

function enlarge_step(
    O::AbstractTensor4{U},
    AL::Tensor3{T},
    AR::Tensor3{T},
    AC::Tensor3{T},
    C::Matrix{T};
    maxenlarge = size(C, 1) * (size(AC, 3) - 1),
    ϵ = 0.0,
) where {T,U}

    V = promote_type(T, U)
    d, p = size(C, 1), size(AC, 3)
    VL, VR = vl_and_vr(AL, AR)
    N = twosite_variance(AL, AR, VL, VR, O)
    x, y, z = svd(N)
    enlarge_d = min(sum(y .> ϵ), maxenlarge)
    enlarge_L = mul_matrix_from_right(VL, x[:, 1:enlarge_d])
    enlarge_R = mul_matrix_from_left(VR, z[:, 1:enlarge_d]')

    AL_new, AR_new, AC_new, C_new = zeros(V, d + enlarge_d, d + enlarge_d, p),
    zeros(V, d + enlarge_d, d + enlarge_d, p),
    zeros(V, d + enlarge_d, d + enlarge_d, p),
    zeros(V, d + enlarge_d, d + enlarge_d)

    AL_new[1:d, 1:d, :] .= AL
    AR_new[1:d, 1:d, :] .= AR
    AC_new[1:d, 1:d, :] .= AC
    C_new[1:d, 1:d] .= C
    AL_new[1:d, d+1:end, :] .= enlarge_L
    AR_new[d+1:end, 1:d, :] .= enlarge_R

    (AL_new, AR_new, AC_new, C_new), (norm(y), norm(y[enlarge_d+1:end]))
end
