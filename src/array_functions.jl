function similar_normalized_eye(C::T) where {T<:AbstractMatrix}
    @assert size(C, 1) == size(C, 2)
    T(Diagonal(I / sqrt(size(C, 1)), size(C, 1)))
end

function _tensor_to_matrix(X::AbstractTensor3, lp_r::Bool = true)
    d = size(X, 1)
    lp_r ? reshape(permutedims(X, (1, 3, 2)), :, d) : reshape(X, d, :)
end

function _matrix_to_tensor(X::AbstractMatrix, lp_r::Bool = true)
    d = lp_r ? size(X, 2) : size(X, 1)
    lp_r ? permutedims(reshape(X, d, :, d), (1, 3, 2)) : reshape(X, d, d, :)
end

"""
    al_and_ar(AC, C)

Compute {AL, AR} which satisfies `mul_matrix_from_right(AL, C) ≈ AC` and
`mul_matrix_from_left(AR, C) ≈ AC`

### Return values:
`(AL, AR), (ϵL, ϵR) = al_and_ar(AC, C)`

where
`ϵL = norm(mul_matrix_from_right(AL, C) - AC)` and
`ϵR = norm(mul_matrix_from_left(AR, C) - AC)`
"""
function al_and_ar(AC::AbstractTensor3, C::AbstractMatrix)
    AC_Cdag = mul_matrix_from_right(AC, C')
    x, y, z = svd(_tensor_to_matrix(AC_Cdag, true))
    AL = _matrix_to_tensor(x * z', true)
    ϵL = norm(AC - mul_matrix_from_right(AL, C))

    Cdag_AC = mul_matrix_from_left(AC, C')
    x, y, z = svd(_tensor_to_matrix(AC_Cdag, false))
    AR = _matrix_to_tensor(x * z', false)
    ϵR = norm(AC - mul_matrix_from_left(AR, C))

    (AL, AR), (ϵL, ϵR)
end

"""
    mixed_canonical(A)

convert non-canonical uniform MPS `|Ψ(A)⟩ = ∑ vₗ^† (∏ᵢ A^{sᵢ}) vᵣ |{s}⟩` to
mixed-canonical uniform MPS
`|Ψ(AL, AR, AC, C)⟩ = ∑ vₗ^† (∏ᵢ AL^{sᵢ}) AC^{sⱼ} (∏ᵢ AR^{sᵢ}) vᵣ |{s}⟩ = ∑ vₗ^† (∏ᵢ AL^{sᵢ}) C (∏ᵢ AR^{sᵢ}) vᵣ |{s}⟩`

### Return values:
`AL, AR, AC, C = mixed_canonical(A)`
"""
function mixed_canonical(A::Tensor3{T}) where T

    d = size(A, 1)

    (L_val,), (L_vec,), L_inf = eigsolve(randn(T, d, d)) do X
        transfer_from_left(X, A)
    end
    (R_val,), (R_vec,), R_inf = eigsolve(randn(T, d, d)) do X
        transfer_from_right(X, A)
    end
    L_vec .*= sign(sum(L_vec))
    R_vec .*= sign(sum(R_vec))
    sqrtL, sqrtR = T.(sqrt(L_vec)), T.(sqrt(R_vec))
    AL = mul_matrix_from_left(mul_matrix_from_right(A, inv(sqrtL)), sqrtL)
    AR = mul_matrix_from_right(mul_matrix_from_left(A, inv(sqrtR)), sqrtR)
    C = sqrtL * sqrtR

    C ./= norm(C)
    AL ./= norm(mul_matrix_from_right(AL, C))
    AR ./= norm(mul_matrix_from_left(AR, C))
    AC = mul_matrix_from_left(AR, C)

    (AL, AR, AC, C)
end