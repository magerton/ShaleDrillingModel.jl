using ShaleDrillingModel
using Base.Test
using StatsFuns
using JLD

jldpath = joinpath(Pkg.dir("ShaleDrillingModel"), "data/price-cost-transitions.jld")
@load jldpath pspace cspace Πpcr Πp1

# some primitives
β = (1.02 / 1.125) ^ (1./12.)  # real discount rate
royalty_rates = [0.125, 1./6., 0.1875, 0.20, 0.225, 0.25]
geology_types = 1:10
royalty_types = 1:length(royalty_rates)

# initial parameters
θt = [1.0, -2.6, -2.4, 1.2]
σv = 0.5

# problem sizes
nψ, dmx, nz, nv =  5, 2, size(Πp1,1), 5
wp = well_problem(dmx,4,10)
zspace, ψspace, dspace, d1space, vspace = (pspace,), linspace(-5.5, 5.5, nψ), 0:dmx, 0:1, linspace(-3.0, 3.0, nv)
# nd, ns, nθ = length(dspace), length(wp), length(θt)

prim = dcdp_primitives(u_add, du_add, duσ_add, β, wp, zspace, Πp1, ψspace, vspace, 1)
tmpv = dcdp_tmpvars(length(θt), prim)
evs = dcdp_Emax(θt, prim)

# check sizes of models
ShaleDrillingModel.check_size(θt, prim, evs)



include("learning_transition.jl")
include("utility.jl")
include("logsumexp3.jl")
include("vf_solve_terminal_and_infill.jl")
include("vf_solve_exploratory.jl")

# check jacobian
check_EVjac(evs, tmpv, prim, θt, σv, 0.2)

include("parallel_solution.jl")

















#
