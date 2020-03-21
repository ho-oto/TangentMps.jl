function similar_normalized_eye(C::T) where {T<:AbstractMatrix}
    @assert size(C, 1) == size(C, 2)
    T(Diagonal(I / sqrt(size(C, 1)), size(C, 1)))
end

function tensor_to_matrix(X::AbstractTensor3, lp_r::Bool = true)
    d = size(X, 1)
    lp_r ? reshape(permutedims(X, (1, 3, 2)), :, d) : reshape(X, d, :)
end

function matrix_to_tensor(X::AbstractMatrix, lp_r::Bool = true)
    d = lp_r ? size(X, 2) : size(X, 1)
    lp_r ? permutedims(reshape(X, d, :, d), (1, 3, 2)) : reshape(X, d, d, :)
end

function al_and_ar(AC::AbstractTensor3, C::AbstractMatrix)
    AC_Cdag = mul_matrix_from_right(AC, C')
    x, y, z = svd(tensor_to_matrix(AC_Cdag, true))
    AL = matrix_to_tensor(x * z', true)
    ϵL = norm(AC - mul_matrix_from_right(AL, C))

    Cdag_AC = mul_matrix_from_left(AC, C')
    x, y, z = svd(tensor_to_matrix(AC_Cdag, false))
    AR = matrix_to_tensor(x * z', false)
    ϵR = norm(AC - mul_matrix_from_left(AR, C))

    (AL, AR), (ϵL, ϵR)
end
