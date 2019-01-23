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
jldpath = "D:/software_libraries/julia/dev/ShaleDrillingData/data/price-vol-transitions.jld"
@load jldpath logp_space logc_space logσ_space Πp Πpc Πpconly

# some primitives
β = (1.02 / 1.125) ^ (1.0/4.0)  # real discount rate
royalty_rates = [0.125, 1.0/6.0, 0.1875, 0.20, 0.225, 0.25]
royalty_types = 1:length(royalty_rates)
geology_types = 1.3430409262656042:0.1925954901417719:5.194950729101042

# initial parameters
flowfuncname = :one_restr
θt = [-4.28566, -5.45746, -0.3, ] # ShaleDrillingModel.STARTING_log_ogip, ShaleDrillingModel.STARTING_σ_ψ,
σv = 0.3

θfull = vcat(θt, σv)

# problem sizes
nψ, dmx, nz, nv =  51, 3, size(Πp,1), 51
# wp = LeasedProblemContsDrill(dmx,4,5,3,2)
# wp = LeasedProblem(dmx,4,5,3,2)
pp = PerpetualProblem(dmx,4,5,3,2)
lp = LeasedProblem(dmx,4,0,-1,0)

roy = 0.2
geoid = 4.0
itype = (geoid, roy,)

zspace, ψspace, dspace, d1space, vspace = (logp_space, logσ_space,), range(-4.5, stop=4.5, length=nψ), 0:dmx, 0:1, range(-3.0, stop=3.0, length=nv)

pprim = dcdp_primitives(flowfuncname, β, pp, zspace, Πp, ψspace)
ptmpv = dcdp_tmpvars(pprim)
pevs = dcdp_Emax(pprim)

lprim = dcdp_primitives(flowfuncname, β, lp, zspace, Πp, ψspace)
ltmpv = dcdp_tmpvars(lprim)
levs = dcdp_Emax(lprim)

# solve - no gradient
solve_vf_all!(pevs, ptmpv, pprim, θt, σv, itype, false)
solve_vf_all!(levs, ltmpv, lprim, θt, σv, itype, false)

ShaleDrillingModel.state_space_vector(pp)
ShaleDrillingModel.state_space_vector(lp)

@test all(pevs.EV .== levs.EV[:,:,[1,3:end...]])

# solve - with gradient
solve_vf_all!(pevs, ptmpv, pprim, θt, σv, itype, true)
solve_vf_all!(levs, ltmpv, lprim, θt, σv, itype, true)

@test all(pevs.EV .== levs.EV[:,:,[1,3:end...]])
@test all(pevs.dEVσ .== levs.dEVσ[:,:,[1,3:end...]])
@test !all(pevs.dEVσ .== 0.0)

# check it out
levs.dEVσ[10,:,:]
pevs.dEVσ[10,:,:]

# ------------------ now do FD check --------------------

let evs = levs,
    tmpv = ltmpv,
    prim = lprim,
    wp = prim.wp

    zero!(evs)

    # now do σ
    h = peturb(σv)
    σ1 = σv - h
    σ2 = σv + h
    hh = σ2 - σ1
    fdEVσ = similar(evs.EV)

    zero!(tmpv)
    solve_vf_all!(evs, tmpv, prim, θt, σ1, itype, false)
    fdEVσ .= -evs.EV

    zero!(tmpv)
    solve_vf_all!(evs, tmpv, prim, θt, σ2, itype, false)
    # fdubv .+= tmpv.ubVfull
    # fdubv ./= hh
    fdEVσ .+= evs.EV
    fdEVσ ./= hh

    zero!(tmpv)
    solve_vf_all!(evs, tmpv, prim, θt, σv, itype, true)

    @views fdEVσvw = fdEVσ[:,:,1:ShaleDrillingModel._nSexp(prim.wp)]
    maxv, idx = findmax(abs.(fdEVσvw.-evs.dEVσ))
    println("worst value is $maxv at $(CartesianIndices(fdEVσvw)[idx]) for dσ")
    @test all(isapprox.(evs.dEVσ, fdEVσvw; atol=1.5e-7))
end
