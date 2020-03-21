const sx2 = [0 0.5; 0.5 0]
const isy2 = [0 0.5; -0.5 0]
const sz2 = [0.5 0; 0 -0.5]
const sp2 = sx2 + isy2
const sm2 = sx2 - isy2
const id2 = Float64[1 0; 0 1]

const sx3 = Float64[0 1 0; 1 0 1; 0 1 0] ./ sqrt(2)
const isy3 = Float64[0 1 0; -1 0 1; 0 -1 0] ./ sqrt(2)
const sz3 = Float64[1 0 0; 0 0 0; 0 0 -1]
const sp3 = sx3 + isy3
const sm3 = sx3 - isy3
const id3 = Float64[1 0 0; 0 1 0; 0 0 1]

const sdots = (kron(sp3, sm3) + kron(sm3, sp3)) / 2 + kron(sz3, sz3)
const blbq(θ) = reshape(cos(θ) .* sdots + sin(θ) .* sdots^2, 3, 3, 3, 3)