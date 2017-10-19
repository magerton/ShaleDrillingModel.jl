# using ShaleDrillingModel
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
nψ, dmx, nz, nv =  5, 2, size(Πp1,1), 5
wp = well_problem(dmx,4,10)
zspace, ψspace, dspace, d1space, vspace = (pspace,), linspace(-5.5, 5.5, nψ), 0:dmx, 0:1, linspace(-3.0, 3.0, nv)
nd, ns, nθ = length(dspace), length(wp), length(θt)


fa = ShaleDrillingModel.u_add
dfa = ShaleDrillingModel.du_add
dfσa = ShaleDrillingModel.duσ_add

prim = dcdp_primitives(fa, dfa, dfσa, β, wp, zspace, Πp1, nψ, vspace, 1)
tmpvars = dcdp_tmpvars(nθ, prim)
evs = dcdp_Emax(θt, prim)

ShaleDrillingModel.check_size(θt, prim, evs)
size(evs.EV)
size(evs.dEV)
size(evs.dEV_σ)

check_flowgrad(θt, σv, prim, extrema(ψspace), 0.2)
evs.EV .= 0.
evs.dEV .= 0.
evs.dEV_σ .= 0.
ShaleDrillingModel.solve_vf_all!(evs, tmpvars, θt .+ rand(length(θt)), σv .+ rand(), prim, extrema(ψspace), 0.2, false)

# check_dEV!(evs, tmpvars, θt, σv, prim, extrema(ψspace), 0.2)
# check_dEV(θt, σv, prim, extrema(ψspace), 0.2)
check_dEVσ(evs, tmpvars, θt, σv, prim, extrema(ψspace), 0.2)


T = eltype(θt)
p = prim
tmp = tmpvars
ψextrema = extrema(ψspace)
roy = 0.2

EV1 = zeros(T, size(evs.EV))
EV2 = zeros(T, size(evs.EV))

h = max( abs(σv), one(T) ) * cbrt(eps(T))
σp = σv + h
σm = σv - h
hh = σp - σm

update_payoffs!(tmp, θt, σv, p, ψextrema, roy, false; h=-h)
solve_vf_all!(EV1, tmp.uin, tmp.uex, tmp.ubVfull, tmp.lse, tmp.tmp, tmp.IminusTEVp, p.wp, p.Πz, tmp.βΠψ, p.β)

update_payoffs!(tmp, θt, σv, p, ψextrema, roy, false; h=h)
solve_vf_all!(EV2, tmp.uin, tmp.uex, tmp.ubVfull, tmp.lse, tmp.tmp, tmp.IminusTEVp, p.wp, p.Πz, tmp.βΠψ, p.β)

solve_vf_all!(evs, tmp, θt, σv, p, ψextrema, roy, true)

dEVk = @view(evs.dEV_σ[:,:,1,1:end-1])
EV1vw = @view(EV1[:,:,1:size(dEVk,3)])
EV2vw = @view(EV2[:,:,1:size(dEVk,3)])
EVfd = (EV2vw .- EV1vw) ./ hh
EVfd ≈ dEVk  ||  warn("Bad grad for σ at vspace[1]")

absd = maximum( absdiff.(EVfd, dEVk ) )
reld = maximum( reldiff.(EVfd, dEVk ) )
println("For σ, abs diff is $absd. max rel diff is $reld")


dEVk























#
