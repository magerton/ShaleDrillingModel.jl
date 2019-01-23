# detect if using SLURM
const IN_SLURM = "SLURM_JOBID" in keys(ENV)

using Distributed
IN_SLURM && using ClusterManagers

using ShaleDrillingModel
# using ShaleDrillingData
using Test
using StatsFuns
using JLD2
using FileIO
using Interpolations
using Statistics
using SparseArrays
using Calculus

# jldpath = Base.joinpath(Pkg.dir("ShaleDrillingData"), "data/price-transitions.jld")
# jldpath = joinpath(dirname(pathof(ShaleDrillingData)), "..", "data/price-vol-transitions.jld")
jldpath = "D:/libraries/julia/dev/ShaleDrillingData/data/price-vol-transitions.jld"
@load jldpath logp_space logc_space logσ_space Πp Πpc Πpconly

# some primitives
β = (1.02 / 1.125) ^ (1.0/4.0)  # real discount rate
royalty_rates = [0.125, 1.0/6.0, 0.1875, 0.20, 0.225, 0.25]
royalty_types = 1:length(royalty_rates)
geology_types = 1.3430409262656042:0.1925954901417719:5.194950729101042

# initial parameters
flowfuncname = :one_restr
θt = [-4.28566, -5.45746, -0.3, ] # ShaleDrillingModel.STARTING_log_ogip, ShaleDrillingModel.STARTING_σ_ψ,
σv = 0.1

θfull = vcat(θt, σv)

# problem sizes
nψ, dmx, nz, nv =  51, 3, size(Πp,1), 51
# wp = LeasedProblemContsDrill(dmx,4,5,3,2)
# wp = LeasedProblem(dmx,4,5,3,2)
# wp = PerpetualProblem(dmx,4,5,3,2)
wp = LeasedProblem(dmx,4,1,-1,0)


# pp = PerpetualProblem(dmx,4,5,3,2)
# lp = LeasedProblem(dmx,4,5,3,2)
# sspp = ShaleDrillingModel.state_space_vector(pp)
# sslp = ShaleDrillingModel.state_space_vector(lp)
#
# ShaleDrillingModel.exploratory_learning(pp)
# ShaleDrillingModel.exploratory_learning(lp)
#
# sspp[ShaleDrillingModel.exploratory_learning(pp)]
# sslp[ShaleDrillingModel.exploratory_learning(lp)]
#
# ShaleDrillingModel.ind_exp(pp)
# ShaleDrillingModel.ind_exp(lp)
#
# sprime(pp,first(ShaleDrillingModel.ind_exp(pp)),0)
# sprime(lp,first(ShaleDrillingModel.ind_exp(lp)),0)
#
#
# wp = LeasedProblem(dmx,4,5,3,2)
# end_ex0(wp)





zspace, ψspace, dspace, d1space, vspace = (logp_space, logσ_space,), range(-4.5, stop=4.5, length=nψ), 0:dmx, 0:1, range(-3.0, stop=3.0, length=nv)

prim = dcdp_primitives(flowfuncname, β, wp, zspace, Πp, ψspace)
tmpv = dcdp_tmpvars(prim)
evs = dcdp_Emax(prim)

## check sizes of models
ShaleDrillingModel.check_size(prim, evs)

# include("BSplineTestFuns_runtests.jl")
include("state-space.jl")
# include("flow-payoffs.jl")

@testset  "testing flow gradients" begin
    let geoid = 2, roy = 0.2
        @test check_flowgrad(θt, σv, prim, geoid, roy)
    end
    @test check_dΠψ(σv, ψspace)
end

# include("test_utility.jl")
include("test_transition.jl")

include("logsumexp3.jl")

include("vf_solve_terminal_and_infill.jl")
include("vf_solve_exploratory.jl")

zero!(tmpv)
let geoid = 2, roy = 0.25, itype = (geoid, roy,)
    solve_vf_all!(evs, tmpv, prim, θt, σv, itype, true)
end

include("vf_interpolation.jl")
include("parallel_solution.jl")
include("action_probabilities_new.jl")
