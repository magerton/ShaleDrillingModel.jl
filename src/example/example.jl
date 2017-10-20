using ShaleDrillingModel
using Base.Test
using StatsFuns
using JLD

pth = joinpath(Pkg.dir("DynamicDrillingProblemEstimation"), "data/price-cost-transitions.jld")
@load pth pspace cspace Πpcr Πp1

# some primitives
β = (1.02 / 1.125) ^ (1./12.)  # real discount rate
royalty_rates = [0.125, 1./6., 0.1875, 0.20, 0.225, 0.25]
geology_types = 1:10
royalty_types = 1:length(royalty_rates)

# initial parameters
θt = [1.0, -2.6, -2.4, 1.2]
σv = 0.5

# problem sizes
wp = well_problem(2, 4, 10)  # max 2 wells/period, max 4 total, 10 periods in lease
nψ, nv =  5, 5
zspace, ψspace, vspace = (pspace,), linspace(-5.5, 5.5, nψ),  linspace(-3.0, 3.0, nv)

# set up primitives. using u_add, du_add, and duσ_add as payoffs + jacobians
prim = dcdp_primitives(u_add, du_add, duσ_add, β, wp, zspace, Πp1, ψspace, vspace, 1)
tmpv = dcdp_tmpvars(length(θt), prim)
evs = dcdp_Emax(θt, prim)

# check sizes of primitives
ShaleDrillingModel.check_size(θt, prim, evs)

# check flow gradients
check_flowgrad(θt, σv, prim, 0.2)

# check jacobian of Emax
check_EVjac(evs, tmpv, prim, θt, σv, 0.2)

# example solver we can do parallelized...
solve_vf_all!(evs, tmpv, prim, vcat(θt, σv), 0.2, 1, Val{true})

# how we do this with globals
set_g_dcdp_primitives(prim)
set_g_dcdp_Emax(evs)
set_g_dcdp_tmpvars(tmpv)

solve_vf_all!(vcat(θt, σv), 0.2, 1, Val{true})
