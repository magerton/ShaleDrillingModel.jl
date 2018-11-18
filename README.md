# ShaleDrillingModel.jl

<!-- [![Build Status](https://travis-ci.org/magerton/ShaleDrillingModel.jl.svg?branch=master)](https://travis-ci.org/magerton/ShaleDrillingModel.jl)

[![Coverage Status](https://coveralls.io/repos/magerton/ShaleDrillingModel.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/magerton/ShaleDrillingModel.jl?branch=master)

[![codecov.io](http://codecov.io/github/magerton/ShaleDrillingModel.jl/coverage.svg?branch=master)](http://codecov.io/github/magerton/ShaleDrillingModel.jl?branch=master) -->


# Example

To get the package & everything needed, run
```julia
Pkg.clone("https://github.com/magerton/GenGlobal.jl", "GenGlobal")
Pkg.clone("https://github.com/magerton/ShaleDrillingModel.jl.git", "ShaleDrillingModel")
Pkg.add("MarkovTransitionMatrices")
Pkg.checkout("MarkovTransitionMatrices", branch="new-moment-matching")
```

See [example/example.jl](example/example.jl) for how to use.

Create price/cost transition matrices using [MarkovTransitionMatrices.jl](https://github.com/magerton/MarkovTransitionMatrices.jl). [example/make_big_transition.jl](example/make_big_transition.jl) has an example of doing this. Note that the version of `MarkovTransitionMatrices.jl` being used is probably `new-moment-matching`, not `master`.


```julia
]add AxisAlgorithms BenchmarkTools Calculus CategoricalArrays DataFrames Distributions FileIO Formatting GLM GR Gadfly IndirectArrays Interpolations JLD2 MixedModels NLSolversBase NLopt Optim Plots Primes Profile ProgressMeter PyPlot RData Ratios StatsBase StatsFuns StatsModels

dev ssh://git@github.com/magerton/CountPlus.git
dev ssh://git@github.com/magerton/Halton.git
dev ssh://git@github.com/magerton/JuliaTex.jl.git
dev ssh://git@github.com/magerton/GenGlobal.jl.git
dev ssh://git@github.com/magerton/MarksRandomEffects.git
dev ssh://git@github.com/magerton/OrderedResponse.jl.git
dev ssh://git@github.com/magerton/MarkovTransitionMatrices.jl.git
dev ssh://git@github.com/magerton/ShaleDrillingModel.jl.git
dev ssh://git@github.com/magerton/ShaleDrillingData.jl.git
dev ssh://git@github.com/magerton/ShaleDrillingEstimation.jl.git
dev ssh://git@github.com/magerton/ShaleDrillingPostEstimation.jl.git
```


Alternatively,

```julia
dev D:/libraries/julia/dev/CountPlus
dev D:/libraries/julia/dev/Halton
dev D:/libraries/julia/dev/JuliaTex
dev D:/libraries/julia/dev/GenGlobal
dev D:/libraries/julia/dev/MarksRandomEffects
dev D:/libraries/julia/dev/OrderedResponse
dev D:/libraries/julia/dev/MarkovTransitionMatrices
dev D:/libraries/julia/dev/ShaleDrillingModel
dev D:/libraries/julia/dev/ShaleDrillingData
dev D:/libraries/julia/dev/ShaleDrillingEstimation
dev D:/libraries/julia/dev/ShaleDrillingPostEstimation
```

```julia
dev ~/.julia/dev/CountPlus
dev ~/.julia/dev/Halton
dev ~/.julia/dev/JuliaTex
dev ~/.julia/dev/GenGlobal
dev ~/.julia/dev/MarksRandomEffects
dev ~/.julia/dev/OrderedResponse
dev ~/.julia/dev/MarkovTransitionMatrices
dev ~/.julia/dev/ShaleDrillingModel
dev ~/.julia/dev/ShaleDrillingData
dev ~/.julia/dev/ShaleDrillingEstimation
dev ~/.julia/dev/ShaleDrillingPostEstimation
```
