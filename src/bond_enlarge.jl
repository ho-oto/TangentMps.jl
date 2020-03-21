"""
    twosite_variance(AL, AR, VL, AR, O)

```
 ----AL-----------AR----
|    [OOOOOOOOOOOOO]    |
 -conj(VL)-   -conj(VR)-
           | |
-----------   -----------
```
"""
function twosite_variance(
    AL::AbstractTensor3,
    AR::AbstractTensor3,
    VL::AbstractTensor3,
    VR::AbstractTensor3,
    O::AbstractTensor4,
)

    let (_, _l, _r, _blp, _brp, _klp, _krp, _x, _y, _z) = ntuple(x -> nothing, 10)
        @tensoropt !(_blp, _brp, _klp, _krp) _[_l, _r] :=
            AL[_x, _y, _klp] *
            AR[_y, _z, _krp] *
            conj(VL)[_x, _l, _blp] *
            conj(VR)[_r, _z, _brp] *
            O[_blp, _brp, _klp, _krp]
    end
end


function vl_and_vr(AL::AbstractTensor3, AR::AbstractTensor3)
    P, D = size(AL, 3), size(AL, 2)
    VL = permutedims(reshape(nullspace(tensor_to_matrix(AL, true)'), D, P, :), (1, 3, 2))
    size(VL, 2) == D * (P - 1) || @warn "failed to solve nullspace in VL"

    P, D = size(AR, 3), size(AR, 2)
    VR = reshape(nullspace(tensor_to_matrix(AR, false))', :, D, P)
    size(VR, 1) == D * (P - 1) || @warn "failed to solve nullspace in VR"

    (VL, VR)
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
