module ShaleDrillingModel

using StatsFuns
using StatsBase
using AxisAlgorithms
using Calculus
using GenGlobal

const AbstractArray3{T} = AbstractArray{T,3}
const AbstractArray4{T} = AbstractArray{T,4}
const AbstractArray5{T} = AbstractArray{T,5}

# package code goes here
include("utility_additive.jl")


include("helpers.jl")
include("learning_transition.jl")
include("makeStateSpace.jl")

include("vf_structs.jl")
include("utility.jl")

include("logsumexp3.jl")

include("makeIminusTVp.jl")

include("vfit.jl")

include("vf_solve_terminal_and_infill.jl")
include("learning_update.jl")
include("vf_solve_exploratory.jl")
include("vf_solve_all.jl")
include("check_dEV.jl")

include("parallel_solution.jl")


end # module
