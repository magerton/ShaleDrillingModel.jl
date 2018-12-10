# detect if using SLURM
const IN_SLURM = "SLURM_JOBID" in keys(ENV)

using Distributed
IN_SLURM && using ClusterManagers

using ShaleDrillingModel
using ShaleDrillingData
using Test
using StatsFuns
using JLD2
using FileIO
using Interpolations
using Statistics
using SparseArrays

# jldpath = Base.joinpath(Pkg.dir("ShaleDrillingData"), "data/price-transitions.jld")
jldpath = joinpath(dirname(pathof(ShaleDrillingData)), "..", "data/price-vol-transitions.jld")
@load jldpath logp_space logc_space logσ_space Πp Πpc

# some primitives
β = (1.02 / 1.125) ^ (1.0/12.0)  # real discount rate
royalty_rates = [0.125, 1.0/6.0, 0.1875, 0.20, 0.225, 0.25]
royalty_types = 1:length(royalty_rates)
geology_types = 1.3430409262656042:0.1925954901417719:5.194950729101042

# initial parameters
flowfuncname = :one
θt = [-4.28566, ShaleDrillingModel.STARTING_log_ogip, ShaleDrillingModel.STARTING_σ_ψ, -5.45746, -0.3, ]
σv = 0.0

θfull = vcat(θt, σv)

# problem sizes
nψ, dmx, nz, nv =  51, 3, size(Πp,1), 51
wp = well_problem(dmx,4,5,3,2)

zspace, ψspace, dspace, d1space, vspace = (logp_space, logσ_space,), range(-3.75, stop=3.75, length=nψ), 0:dmx, 0:1, range(-3.0, stop=3.0, length=nv)

prim = dcdp_primitives(flowfuncname, β, wp, zspace, Πp, ψspace)
tmpv = dcdp_tmpvars(prim)
evs = dcdp_Emax(prim)

## check sizes of models
ShaleDrillingModel.check_size(prim, evs)

# include("makeStateSpace.jl")
# include("flow-payoffs.jl")
#
# @testset  "testing flow gradients" begin
#     let geoid = 2, roy = 0.2
#         @test check_flowgrad(θt, σv, prim, geoid, roy)
#     end
#     @test check_dΠψ(σv, ψspace)
# end
#
# include("test_utility.jl")
# include("test_transition.jl")
#
# println("filling per-period payoffs")
#
# let geoid = 2, roy = 0.25, itype = (geoid, roy,)
#     @views fillflows!(flow(prim), flow, tmpv.uin[:,:,:,   1], tmpv.uin[:,:,:,   2], tmpv.uex, θt, σv, makepdct(prim, θt, Val{:u},  σv), itype...)
#     fillflows_grad!(tmpv, prim, θt, σv, itype...)
# end
#
# include("logsumexp3.jl")
#
# include("vf_solve_terminal_and_infill.jl")
# include("vf_solve_exploratory.jl")
#
# zero!(tmpv)
# let geoid = 2, roy = 0.25, itype = (geoid, roy,)
#     solve_vf_all!(evs, tmpv, prim, θt, σv, itype, Val{true})
# end
#
# include("vf_interpolation.jl")
# include("parallel_solution.jl")
include("action_probabilities_new.jl")
include("BSplineTestFuns_runtests.jl")

# include("test_dpsi.jl") DEPRECATED & INCORRECT???
