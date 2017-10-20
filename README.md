# ShaleDrillingModel.jl

[![Build Status](https://travis-ci.org/magerton/ShaleDrillingModel.jl.svg?branch=master)](https://travis-ci.org/magerton/ShaleDrillingModel.jl)

[![Coverage Status](https://coveralls.io/repos/magerton/ShaleDrillingModel.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/magerton/ShaleDrillingModel.jl?branch=master)

[![codecov.io](http://codecov.io/github/magerton/ShaleDrillingModel.jl/coverage.svg?branch=master)](http://codecov.io/github/magerton/ShaleDrillingModel.jl?branch=master)


# Example

To get the package, run
```julia
Pkg.clone("https://github.com/magerton/ShaleDrillingModel.jl.git")
```

See [example/example.jl](example/example.jl) for how to use.

Create price/cost transition matrices using [MarkovTransitionMatrices.jl](https://github.com/magerton/MarkovTransitionMatrices.jl). [example/make_big_transition.jl](example/make_big_transition.jl) has an example of doing this. Note that the version of `MarkovTransitionMatrices.jl` being used is probably `new-moment-matching`, not `master`.
