module ShaleDrillingModel

using StatsFuns
using AxisAlgorithms
using Calculus

const AbstractArray3{T} = AbstractArray{T,3}
const AbstractArray4{T} = AbstractArray{T,4}
const AbstractArray5{T} = AbstractArray{T,5}

# package code goes here

include("helpers.jl")
include("tauchen86.jl")
include("logsumexp3.jl")
include("makeStateSpace.jl")
include("utility.jl")
include("makeIminusTVp.jl")
include("vfit_learning.jl")
include("vf_total_solve_learning.jl")

end # module
