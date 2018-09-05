module ShaleDrillingModel

using StatsFuns
using StatsBase
using AxisAlgorithms
using Calculus
using GenGlobal
using Interpolations

using SharedArrays
using SparseArrays
using Distributed
using LinearAlgebra


const AbstractArray3{T} = AbstractArray{T,3}
const AbstractArray4{T} = AbstractArray{T,4}
const AbstractArray5{T} = AbstractArray{T,5}

include("helpers.jl")
include("learning_transition.jl")
include("makeStateSpace.jl")

# package code goes here
include("flow-payoffs.jl")

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
include("BSplineExtensions.jl")

include("vf_interpolation.jl")

include("action_probabilities.jl")
include("prob_drill_and_landowner_rev.jl")


# end module
end
