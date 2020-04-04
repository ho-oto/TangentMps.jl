"""
    twosite_variance(AL, AC, VL, VR, O)

```
  ┌────────┐    ┌────────┐
┌─1   AL   2────1   AC   2─┐
│ └────3───┘    └───3────┘ │
│      │            │      │
│    .─3────────────4──.   │
│   (         O         )  │
│    `─1────────────2──'   │
│      │            │      │
│ ┌────3───┐    ┌───3────┐ │
└─1conj(VL)2┐  ┌1conj(VR)2─┘
  └────────┘│  │└────────┘
(1)─────────┘  └─────────(2)
```
"""
function twosite_variance(
    AL::AbstractTensor3,
    AC::AbstractTensor3,
    VL::AbstractTensor3,
    VR::AbstractTensor3,
    O::AbstractTensor4,
)
    @tensoropt ((blp, brp, klp, krp) = 1, (x, y, z, l, r) = χ) twosite_variance[l, r] :=
        AL[x, y, klp] *
        AC[y, z, krp] *
        conj(VL)[x, l, blp] *
        conj(VR)[r, z, brp] *
        O[blp, brp, klp, krp]
end
function twosite_variance(
    AL::AbstractTensor3,
    AC::AbstractTensor3,
    VL::AbstractTensor3,
    O::AbstractTensor4,
)
    @tensoropt ((blp, brp, klp, krp) = 1, (x, y, z, l, r) = χ) twosite_variance[l, r] :=
        AL[x, y, klp] *
        AC[y, z, krp] *
        conj(VL)[x, l, blp] *
        conj(VL)[z, r, brp] * # VR = permutedims(VL, (2, 1, 3))
        O[blp, brp, klp, krp]
end

function vl_from_al(AL::AbstractTensor3)
    P, D = size(AL, 3), size(AL, 2)
    VL = permutedims(reshape(nullspace(_tensor_to_matrix(AL, true)'), D, P, :), (1, 3, 2))
    size(VL, 2) == D * (P - 1) || @warn "failed to solve nullspace in VL"
    VL
end
function vr_from_ar(AR::AbstractTensor3)
    P, D = size(AR, 3), size(AR, 2)
    VR = reshape(nullspace(_tensor_to_matrix(AR, false))', :, D, P)
    size(VR, 1) == D * (P - 1) || @warn "failed to solve nullspace in VR"
    AR
end

function enlargestep(
    TV::AbstractMatrix{U},
    AL::Tensor3{T},
    AR::Tensor3{T},
    AC::Tensor3{T},
    C::Matrix{T},
    VL::Tensor3{T},
    VR::Tensor3{T};
    maxenlarge = size(C, 1) * (size(AC, 3) - 1),
    ϵ = 0.0,
) where {T,U}

    W = promote_type(T, U)
    d, p = size(C, 1), size(AC, 3)

    x, y, z = svd(TV)
    enlarge_d = min(sum(y .> ϵ), maxenlarge)
    enlarge_L = mul_matrix_from_right(VL, x[:, 1:enlarge_d])
    enlarge_R = mul_matrix_from_left(VR, z[:, 1:enlarge_d]')

    AL_new, AR_new, AC_new, C_new = zeros(W, d + enlarge_d, d + enlarge_d, p),
    zeros(W, d + enlarge_d, d + enlarge_d, p),
    zeros(W, d + enlarge_d, d + enlarge_d, p),
    zeros(W, d + enlarge_d, d + enlarge_d)

    AL_new[1:d, 1:d, :] .= AL
    AR_new[1:d, 1:d, :] .= AR
    AC_new[1:d, 1:d, :] .= AC
    C_new[1:d, 1:d] .= C
    AL_new[1:d, d+1:end, :] .= enlarge_L
    AR_new[d+1:end, 1:d, :] .= enlarge_R

    (al = AL_new, ar = AR_new, ac = AC_new, c = C_new), (norm(y), norm(y[enlarge_d+1:end]))
end
function enlargestep(
    TV::AbstractMatrix{U},
    AL::Tensor3{T},
    AC::Tensor3{T},
    C::Matrix{T},
    VL::Tensor3{T};
    maxenlarge = size(C, 1) * (size(AC, 3) - 1),
    ϵ = 0.0,
) where {T,U}

    W = promote_type(T, U)
    d, p = size(C, 1), size(AC, 3)
    VR = permutedims(VL, (2, 1, 3))

    x, y, z = svd(TV)
    enlarge_d = min(sum(y .> ϵ), maxenlarge)
    enlarge_L = mul_matrix_from_right(VL, x[:, 1:enlarge_d])

    AL_new, AC_new, C_new = zeros(W, d + enlarge_d, d + enlarge_d, p),
    zeros(W, d + enlarge_d, d + enlarge_d, p),
    zeros(W, d + enlarge_d, d + enlarge_d)

    AL_new[1:d, 1:d, :] .= AL
    AC_new[1:d, 1:d, :] .= AC
    C_new[1:d, 1:d] .= C
    AL_new[1:d, d+1:end, :] .= enlarge_L

    (al = AL_new, ac = AC_new, c = C_new), (norm(y), norm(y[enlarge_d+1:end]))
end
