module ShaleDrillingModel

using StatsFuns
using AxisAlgorithms
using Calculus

# package code goes here

include("tauchen86.jl")
include("logsumexp3.jl")
include("makeStateSpace.jl")
include("utility.jl")
include("makeIminusTVp.jl")
include("vfit_learning.jl")
include("vf_total_solve_learning.jl")

end # module
