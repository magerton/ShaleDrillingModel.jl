# detect if using SLURM
const IN_SLURM = "SLURM_JOBID" in keys(ENV)

using Distributed
IN_SLURM && using ClusterManagers

using ShaleDrillingModel
using Test
using StatsFuns
using Interpolations
using SparseArrays
using Calculus
using MarkovTransitionMatrices

# -------------------------------------------------
# price transition matrices
# -------------------------------------------------

nlogp = 15
nlogc = 11
nlogσ = 9

extrema_logp = [0.8510954,  2.405270]
extrema_logc = [5.0805023,  5.318393]
extrema_logσ = [-2.9558226, -1.542508]
sdlogσ = 0.18667531

Σpc = [0x1.4a1ecb9bcddfep-7  0x1.94e52859805cap-10;
       0x1.94e52859805cap-10 0x1.4cf39bf87b735p-9]

# make range for coefs
logp_space = range(extrema_logp[1] - log(2.0), stop=extrema_logp[2] + log(2.0), length=nlogp)
logc_space = range(extrema_logc[1] - log(2.0), stop=extrema_logc[2] + log(2.0), length=nlogc)
logσ_space = range(extrema_logσ[1] - 2*sdlogσ, stop=extrema_logσ[2] + 2*sdlogσ, length=nlogσ)

P_σ() = tauchen_1d(logσ_space, (x) -> x, sdlogσ)
P_pricecost() = tauchen_2d(Base.Iterators.product(logp_space, logc_space), (x) -> x, Σpc)
P_price() = tauchen_1d(logp_space, (x) -> x, Σpc[1,1])

# range for i,j block in matrix where block is n x n
blockrange(i,j,n) = ((i-1)*n+1):(i*n),  ((j-1)*n+1):(j*n)

function P_pricevol()
    Pσ = P_σ()
    P  = Matrix{eltype(logp_space)}(undef, nlogp*nlogσ, nlogp*nlogσ)
    for (j, logσj) in enumerate(logσ_space)
      for (i, logσi) in enumerate(logσ_space)
        P[blockrange(i,j,nlogp)...]        .= Pσ[i, j] .* tauchen_1d(logp_space, (x) -> x, exp(logσi)^2)
      end
    end
    return P
end
function P_pricecostvol()
    Pσ = P_σ()
    P = Matrix{eltype(logp_space)}(undef, nlogp*nlogc*nlogσ, nlogp*nlogc*nlogσ)
    for (j, logσj) in enumerate(logσ_space)
      for (i, logσi) in enumerate(logσ_space)
        P[blockrange(i,j,nlogp*nlogc)...] .= Pσ[i, j] .* tauchen_2d(Base.Iterators.product(logp_space, logc_space), (x) -> x, Σpc*exp(logσi))
      end
    end
    return P
end

minp = 1e-4
Πp      = MarkovTransitionMatrices.sparsify!(P_pricevol(),     minp)
Πpc     = MarkovTransitionMatrices.sparsify!(P_pricecostvol(), minp)
Πpconly = MarkovTransitionMatrices.sparsify!(P_pricecost(),    minp)

# -------------------------------------------------
# price transition matrices
# -------------------------------------------------

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
wp = LeasedProblem(dmx,4,5,3,2)
# wp = PerpetualProblem(dmx,4,5,3,2)
# lp = LeasedProblem(dmx,4,1,-1,0)

zspace, ψspace, dspace, d1space, vspace = (logp_space, logσ_space,), range(-4.5, stop=4.5, length=nψ), 0:dmx, 0:1, range(-3.0, stop=3.0, length=nv)

prim = dcdp_primitives(flowfuncname, β, wp, zspace, Πp, ψspace)
tmpv = dcdp_tmpvars(prim)
evs = dcdp_Emax(prim)

## check sizes of models
ShaleDrillingModel.check_size(prim, evs)

include("BSplineTestFuns_runtests.jl")
include("state-space.jl")
# include("flow-payoffs.jl")

@testset  "testing flow gradients" begin
    let geoid = 2, roy = 0.2
        @test check_flowgrad(θt, σv, prim, geoid, roy)
    end
    @test check_dΠψ(σv, ψspace)
end

include("test_utility.jl")
include("test_transition.jl")

include("logsumexp3.jl")
include("vfit.jl")
include("vf_solve_terminal_and_infill.jl")
include("vf_solve_exploratory.jl")

zero!(tmpv)
let geoid = 2, roy = 0.25, itype = (geoid, roy,)
    solve_vf_all!(evs, tmpv, prim, θt, σv, itype, true)
end

include("vf_interpolation.jl")
include("parallel_solution.jl")
include("action_probabilities_new.jl")
