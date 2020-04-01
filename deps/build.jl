using Pkg
Pkg.rm("TensorOperations")
Pkg.add(Pkg.PackageSpec(name="TensorOperations", url="https://github.com/ho-oto/TensorOperations.jl", rev="master"))