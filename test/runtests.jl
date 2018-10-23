using ShaleDrillingModel
using Base.Test
using StatsFuns
using JLD
using Interpolations

include("makeStateSpace.jl")

jldpath = joinpath(Pkg.dir("ShaleDrillingData"), "data/price-cost-transitions.jld")
@load jldpath pspace cspace Πpcr Πp1

# some primitives
β = (1.02 / 1.125) ^ (1./12.)  # real discount rate
royalty_rates = [0.125, 1./6., 0.1875, 0.20, 0.225, 0.25]
geology_types = 1:10
royalty_types = 1:length(royalty_rates)

# initial parameters
θt = [-2.0, 0.932, 1.0, -2.11996, -0.818109, 1.85105,]
σv = 0.566889
θfull = vcat(θt, σv)

# problem sizes
nψ, dmx, nz, nv, ngeo =  51, 3, size(Πp1,1), 51, 1
wp = well_problem(dmx,4,10)
zspace, ψspace, dspace, d1space, vspace = (pspace,), linspace(-3.75, 3.75, nψ), 0:dmx, 0:1, linspace(-3.0, 3.0, nv)

prim = dcdp_primitives(:lin, β, wp, zspace, Πp1, ψspace)
tmpv = dcdp_tmpvars(prim)
evs = dcdp_Emax(prim)

# check sizes of models
ShaleDrillingModel.check_size(prim, evs)

println("testing flow payoffs")
include("flow-payoffs.jl")


# println("testing flow gradients")
# @test check_flowgrad(θt, σv, prim, 0.2, 1)
# println("testing transition derivatives")
# @test check_dΠψ(σv, ψspace)
#
# include("test_utility.jl")
# include("test_transition.jl")
#
#
# println("filling per-period payoffs")
# @views fillflows!(flow(prim), flow, tmpv.uin[:,:,:,   1], tmpv.uin[:,:,:,   2],  tmpv.uex, θt, σv, makepdct(prim, θt, Val{:u},  σv), 0.25, 1)
# fillflows_grad!(tmpv, prim, θt, σv, 0.2, 1)
#
# include("logsumexp3.jl")
#
# include("vf_solve_terminal_and_infill.jl")
# include("vf_solve_exploratory.jl")
#
# zero!(tmpv)
# solve_vf_all!(evs, tmpv, prim, θt, σv, (0.2, 1), Val{true})
#
# include("vf_interpolation.jl")
#
# include("test_dpsi.jl")
# include("parallel_solution.jl")
#
# include("action_probabilities_new.jl")
#
#











#
