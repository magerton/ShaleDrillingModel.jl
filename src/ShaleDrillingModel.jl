module ShaleDrillingModel

using StatsFuns
using AxisAlgorithms
using Calculus

const AbstractArray3{T} = AbstractArray{T,3}
const AbstractArray4{T} = AbstractArray{T,4}
const AbstractArray5{T} = AbstractArray{T,5}

# package code goes here
include("utility_additive.jl")


include("helpers.jl")
include("tauchen86.jl")
include("logsumexp3.jl")
include("makeStateSpace.jl")

include("vf_structs.jl")

include("utility.jl")
include("makeIminusTVp.jl")
include("vfit.jl")


include("vf_solve_terminal_and_infill.jl")
include("learning_update.jl")
include("vf_solve_exploratory.jl")
include("vf_solve_all.jl")





end # module
