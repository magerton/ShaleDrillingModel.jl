using ShaleDrillingModel
using Base.Test
using StatsFuns
using JLD
using Interpolations

jldpath = joinpath(Pkg.dir("ShaleDrillingModel"), "data/price-cost-transitions.jld")
@load jldpath pspace cspace Πpcr Πp1

# some primitives
β = (1.02 / 1.125) ^ (1./12.)  # real discount rate
royalty_rates = [0.125, 1./6., 0.1875, 0.20, 0.225, 0.25]
geology_types = 1:10
royalty_types = 1:length(royalty_rates)

# initial parameters
θt = [1.0, -2.6, -2.4, 1.2]
σv = 1.0

# problem sizes
nψ, dmx, nz, nv =  101, 2, size(Πp1,1), 51
wp = well_problem(dmx,4,10)
zspace, ψspace, dspace, d1space, vspace = (pspace,), linspace(-6.0, 6.0, nψ), 0:dmx, 0:1, linspace(-3.0, 3.0, nv)
# nd, ns, nθ = length(dspace), length(wp), length(θt)

prim = dcdp_primitives(u_add, du_add, duσ_add, β, wp, zspace, Πp1, ψspace, vspace, 1)
tmpv = dcdp_tmpvars(length(θt), prim)
evs = dcdp_Emax(θt, prim)

# check sizes of models
ShaleDrillingModel.check_size(θt, prim, evs)

if false
    include("learning_transition.jl")
    include("utility.jl")
    include("logsumexp3.jl")
    include("vf_solve_terminal_and_infill.jl")
    include("vf_solve_exploratory.jl")

    for r in royalty_rates
        println("test royalty rate $r")
        check_EVjac(evs, tmpv, prim, θt, σv, r)
    end
    # check jacobian
    include("parallel_solution.jl")
end

shev = SharedEV([1,], vcat(θt, σv), prim, royalty_rates, 1:1)
isev = ItpSharedEV(shev, prim, σv)
set_g_dcdp_primitives(prim)
set_g_dcdp_tmpvars(tmpv)
set_g_SharedEV(shev)

# s = parallel_solve_vf_all!(shev, vcat(θt,σv), Val{true})
# fetch.(s)

z = (pspace[15],)
bslin = BSpline(Linear())

itev = interpolate!(evs.EV, (bslin, bslin, NoInterp()), OnGrid())
sitev = scale(itev, pspace, ψspace, 1:_nS(prim))

nSexp1 = _nSexp(prim)+1
itdevsig = interpolate!(evs.dEV_σ, (bslin, bslin, bslin, NoInterp()), OnGrid())
sitdevsig = scale(itdevsig, pspace, ψspace, vspace, 1:nSexp1)

pdct = Base.product( pspace, vspace, vspace, 1:nSexp1)
dEVσ = Array{Float64}(size(pdct))
EVσ1 = similar(dEVσ)
EVσ2 = similar(dEVσ)
fdEVσ = similar(dEVσ)

h = cbrt(eps(Float64))
σ1 = σv-h
σ2 = σv+h
hh = σ2 - σ1
solve_vf_all!(evs, tmpv, prim, θt, σv, 0.2, true)
@show maximum(abs.(evs.EV)), maximum(abs.(evs.dEV)), maximum(abs.(evs.dEV_σ))
for (i,xi) in enumerate(pdct)
    z, u, v, s = xi
    dEVσ[i] = sitdevsig[z, u+σv*v, v, s]
end
@show maximum(abs.(dEVσ))

solve_vf_all!(evs, tmpv, prim, θt, σ1, 0.2, false)
for (i,xi) in enumerate(pdct)
    z, u, v, s = xi
    EVσ1[i] = sitdevsig[z, u+σ1*v, v, s]
end
ev1 = deepcopy(evs.EV)

solve_vf_all!(evs, tmpv, prim, θt, σ2, 0.2, false)
for (i,xi) in enumerate(pdct)
    z, u, v, s = xi
    EVσ2[i] = sitdevsig[z, u+σ2*v, v, s]
end
ev2 = deepcopy(evs.EV)

(ev2 .- ev1) ./ hh


fdEVσ .= (EVσ2 .- EVσ1) ./ hh
worstval, worsti = findmax(abs.(dEVσ .- fdEVσ))
worstind = ind2sub(dEVσ, worsti)
println("In evaluating FD of EV over u&v, worst absdiff at $worstind = $worstval")
@test worstval < 1e-4
























# include("pre-action-probabilities.jl")
# include("action_probabilities.jl")


#
