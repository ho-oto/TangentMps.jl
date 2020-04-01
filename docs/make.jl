using Documenter, TangentMps

makedocs(;
    modules=[TangentMps],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/ho-oto/TangentMps.jl/blob/{commit}{path}#L{line}",
    sitename="TangentMps.jl",
    authors="Hayate Nakano",
    assets=String[],
)

deploydocs(;
    repo="github.com/ho-oto/TangentMps.jl",
)
