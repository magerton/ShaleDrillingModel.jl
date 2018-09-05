using ShaleDrillingModel
using Test
using StatsFuns
using JLD

# load in prices, costs, and transition between high/low volatility regimes
# see eample/make_big_transition.jl
jldpath = joinpath(Pkg.dir("ShaleDrillingModel"), "data/price-cost-transitions.jld")
@load jldpath pspace cspace Πpcr Πp1

# -------- state space ----------

# set of observable types
royalty_rates = [0.125, 1./6., 0.1875, 0.20, 0.225, 0.25]
geology_types = 1:10

# number of points in unobserved heterogeneity grid
nψ, nv =  5, 5

# grids for prices & unobserved heterogeneity
zspace, ψspace, vspace = (pspace,), range(-5.5, stop=5.5, length=nψ),  range(-3.0, stop=3.0, length=nv)

# Structure of time / number of wells
# drill 0 to 2 wells/period
# can drill up to 4 wells total
# 10+1 periods before lease expires
wp = well_problem(2, 4, 10)

# discount rate
β = (1.02 / 1.125) ^ (1./12.)  # real discount rate

# ------------- set up primitives & tmp vars -------------

# initial parameters
θt = [1.0, -2.6, -2.4, 1.2]  # flow payoff parameters
σv = 0.5                     # unobserved heterogeneity

# primitives holds (almost) all information to solve VF except observable type
# flow payoffs + gradients are u_add, du_add, and duσ_add
prim = dcdp_primitives(u_add, du_add, duσ_add, β, wp, zspace, Πp1, ψspace, vspace, 1)

# temporary variables
tmpv = dcdp_tmpvars(length(θt), prim)

# Emax + gradients
evs = dcdp_Emax(θt, prim)

# check sizes of primitives
ShaleDrillingModel.check_size(θt, prim, evs)

# check flow gradients w/ finite differences
check_flowgrad(θt, σv, prim, 0.2)

# solve problem & check jacobian of Emax w/ finite difference
check_EVjac(evs, tmpv, prim, θt, σv, 0.2)

# ----------- how we do this in parallel ------------

# set up workers
pids = addprocs()

# tell workers about the pkg
@everywhere @show pwd()
@everywhere using ShaleDrillingModel

# initialize struct of shared arrays
sev = SharedEV(pids, vcat(θt, σv), prim, royalty_rates, 1:1)

# send primitives to workers
@eval @everywhere begin
    set_g_dcdp_primitives($prim)
    set_g_dcdp_tmpvars($tmpv)
    set_g_SharedEV($sev)
end

# parallel solve
parallel_solve_vf_all!(sev, vcat(θt,σv), Val{true})

using Plots
gr()

nSexpl = ShaleDrillingModel._nSexp(prim)  # number of states in initial period
nS = ShaleDrillingModel._nS(prim)         # number of endogenous states overall

plot(prim.wp.τmax:-1:0, sev.EV[15, 3, 1:nSexpl, :, 1], xaxis=(:flip,), label=string.(round.(royalty_rates,3)), xlabel="time remaining", ylabel="Emax(V)")
plot(exp.(pspace), sev.EV[:, 3, nSexpl+1:nS, 4, 1], xlabel="Price", ylabel="Emax(V)")
