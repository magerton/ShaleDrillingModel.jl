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
nψ, dmx, nz, nv =  5, 2, size(Πp1,1), 5
wp = well_problem(dmx,4,10)
zspace, ψspace, dspace, d1space, vspace = (pspace,), linspace(-5.5, 5.5, nψ), 0:dmx, 0:1, linspace(-3.0, 3.0, nv)
nd, ns, nθ = length(dspace), length(wp), length(θt)

fa = ShaleDrillingModel.u_add
dfa = ShaleDrillingModel.du_add
dfσa = ShaleDrillingModel.duσ_add

prim = dcdp_primitives(fa, dfa, dfσa, β, wp, zspace, Πp1, ψspace, vspace, 1)
tmpv = dcdp_tmpvars(nθ, prim)
evs = dcdp_Emax(θt, prim)

ShaleDrillingModel.check_size(θt, prim, evs)
size(evs.EV)
size(evs.dEV)
size(evs.dEV_σ)

check_flowgrad(θt, σv, prim, 0.2)
evs.EV .= 0.
evs.dEV .= 0.
evs.dEV_σ .= 0.

# ---------------- check on ∂Πψ/∂σ ------------------

ShaleDrillingModel.check_dπdσ(σv, ψspace, vspace)

let x2 = similar(tmpv.βΠψ),
    x1 = similar(tmpv.βΠψ),
    fd = similar(tmpv.βΠψ),
    d = tmpv.βdΠψ,
    v = vspace[3],
    h =  cbrt(eps(Float64))

    ShaleDrillingModel._fdβΠψ!(x2, ψspace, σv, β, v, h)
    ShaleDrillingModel._fdβΠψ!(x1, ψspace, σv, β, v, -h)
    ShaleDrillingModel._dβΠψ!(d, ψspace, σv, β, v)

    fd .= (x2 .- x1) ./ (2.0 .* h)
    # @show maximum(abs.(fd - d))
    @test fd ≈ d

    ShaleDrillingModel._βΠψ!(tmpv.βΠψ, ψspace, σv, 1.0)
    @test all(sum(tmpv.βΠψ, 2) .≈ 1.0)

    ShaleDrillingModel._fdβΠψ!(x2, ψspace, σv, 1.0, 1.0, cbrt(eps(Float64)))
    @test all(sum(x2, 2) .≈ 1.0)
end


# ------------------------------- show that derivatives for flow-payoffs are correct -----------------------------------

let logp = 1.0, ψ = ψspace[5], v = vspace[1], omroy = 0.8, d1 = 0, Dgt0 = false, d=3
    x = duσ_add(θt, σv, logp, ψ, v, d, omroy)
    @test x ≈ Calculus.derivative((h) -> u_add(     θt, σv+h, logp, ψ+h*v, d, d1, Dgt0, omroy      ), 0.0)
    @test x ≈ Calculus.derivative((h) -> fduσ(u_add, θt, σv,  (logp, ψ, d,),   d1, Dgt0, omroy, v, h), 0.0)
end

let duexσ = similar(tmpv.duexσ), uex1 = similar(tmpv.uex), uex2 = similar(tmpv.uex),
    v = vspace[1], h = cbrt(eps(Float64)), d1 = 0, Dgt0 = false, omroy = 0.8
    @inbounds for (i, st) in enumerate(makepdct(zspace, ψspace, vspace, wp, θt, Val{:u}))
        logp, ψ, d = st
        uex1[i] = u_add( θt, σv-h, logp, ψ-h*v, d, d1, Dgt0, omroy)
        uex2[i] = u_add( θt, σv+h, logp, ψ+h*v, d, d1, Dgt0, omroy)
    end
    @inbounds for (i, st) in enumerate(makepdct(zspace, ψspace, vspace, wp, θt, Val{:duσ}))
        logp, ψ, v, d = st
        duexσ[i] = duσ_add(θt, σv, logp, ψ, v, d, omroy)
    end
    fduσex = (uex2 .- uex1) ./ (2.*h)
    @test fduσex ≈ duexσ[:,:,1,:]
end


let duexσ = similar(tmpv.duexσ),
    uex1 = similar(tmpv.uex),
    uex2 = similar(tmpv.uex),
    uin  = similar(tmpv.uin),
    v = vspace[1], h = cbrt(eps(Float64)), d1 = 0, Dgt0 = false, roy = 0.2,
    pdσ = makepdct(zspace, ψspace, vspace, wp, θt, Val{:duσ}),
    pdex = makepdct(zspace, ψspace, vspace, wp, θt, Val{:u})

    uin0 = @view(uin[:,:,:,1])
    uin1 = @view(uin[:,:,:,2])

    fillflows!(u_add, uin0, uin1, uex1, θt, σv, pdex, roy, v, -h)
    fillflows!(u_add, uin0, uin1, uex2, θt, σv, pdex, roy, v, h)
    fillflows!(duσ_add, duexσ, θt, σv, pdσ, roy)
    fduσex = (uex2 .- uex1) ./ (2.0*h)

    @test fduσex ≈ duexσ[:,:,1,:]
end


let duexσ = similar(tmpv.duexσ),
    uex1  = similar(tmpv.uex),
    uex2  = similar(tmpv.uex),
    fdex  = similar(tmpv.uex),
    vpos = 3,
    h = cbrt(eps(Float64)),
    roy = 0.2, p = prim, t = tmpv, σ = σv

    fillflows!(t, p, θt, σ, roy, vspace[vpos], -h)
    uex1 .= t.uex
    fillflows!(t, p, θt, σ, roy, vspace[vpos], h)
    uex2 .= t.uex
    fillflows!(p.dfσ, t.duexσ, θt, σ, makepdct(p, θt, Val{:duσ}, σ), roy)

    fdex .= (uex2 .- uex1) ./ (2.0.*h)
    vw = @view(t.duexσ[:,:,vpos,:])
    @test fdex ≈ vw
end

# ------------------------------- now compute the infill -----------------------------------

roy = 0.2
evs.EV .= 0.0
fillflows_grad!(tmpv, prim, θt, σv, roy)
solve_vf_terminal!(evs)
solve_vf_infill!(evs, tmpv, prim)

using Plots
gr()

plot(exp.(pspace), evs.EV[:, 3, end-1:-1:end-6])


# ------------------------------- exploratory ----------------------------------

tmpv.ubVfull .= 0.
tmpv.dubVfull .= 0.
tmpv.dubV_σ .= 0.

let fduσbv = zeros(size(tmpv.ubVfull)),
    ubv1   = zeros(size(tmpv.ubVfull)),
    ubv2   = zeros(size(tmpv.ubVfull)),
    uin1   = zeros(size(tmpv.uin)),
    uin1   = zeros(size(tmpv.uin)),
    uin2   = zeros(size(tmpv.uin)),
    uex1   = zeros(size(tmpv.uex)),
    uex2   = zeros(size(tmpv.uex)),
    EV1 = similar(evs.EV),
    EV2 = similar(evs.EV),
    vpos = 1,
    h = cbrt(eps(Float64)),
    t = tmpv,
    p = prim,
    σ = σv,
    fdEV = similar(evs.EV)

    v = vspace[vpos]

    fillflows!(tmpv, p, θt, σ, roy, v, -h)
    solve_vf_infill!(evs, tmpv, prim)
    learningUpdate!(tmpv, evs, prim, σv, false, v, -h)
    uin1 .= tmpv.uin
    uex1 .= tmpv.uex
    solve_vf_explore!(evs.EV, uex1, tmpv.ubVfull, tmpv.lse, tmpv.tmp, p.wp, p.Πz, tmpv.βΠψ, β)
    ubv1 .= tmpv.ubVfull
    EV1 .= evs.EV

    fillflows!(tmpv, p, θt, σ, roy, v, +h)
    solve_vf_infill!(evs, tmpv, prim)
    learningUpdate!(tmpv, evs, prim, σv, false, v, h)
    uin2 .= tmpv.uin
    uex2 .= tmpv.uex
    @test all(uin1 .== uin2)
    @test !all(uex1 .== uex2)
    solve_vf_explore!(evs.EV, uex2, tmpv.ubVfull, tmpv.lse, tmpv.tmp, p.wp, p.Πz, tmpv.βΠψ, β)
    ubv2 .= tmpv.ubVfull
    EV2 .= evs.EV

    fillflows_grad!(tmpv, p, θt, σ, roy)
    @test (uex2 .- uex1) ./ (2.0 .* h) ≈ tmpv.duexσ[:,:,vpos,:]
    solve_vf_infill!(evs, tmpv, prim)
    learningUpdate!(tmpv, evs, prim, σ, true)
    solve_vf_explore!(evs, tmpv, prim)

    fduσbv .= (ubv2 .- ubv1) ./ (2.0.*h)
    @show findmax(abs.(fduσbv .-  tmpv.dubV_σ[:,:,vpos,:]))
    # only test 2:end b/c we don't update ubV at the end...
    @test tmpv.dubV_σ[:,:,vpos, 2:end] ≈ fduσbv[:,:,2:end]

    fdEV .= (EV2 .- EV1) ./ (2.0.*h)
    fdEVvw = @view(fdEV[:,:,explore_state_inds(wp)[end:-1:1]])
    dEVσvw = @view(evs.dEV_σ[:,:,vpos,1:end-1])
    @test fdEVvw ≈ dEVσvw
end

# -----------------------------------------------------------------






#
#
#
#
# ShaleDrillingModel.solve_vf_all!(evs, tmpvars, θt .+ rand(length(θt)), σv .+ rand(), prim, extrema(ψspace), 0.2, false)
#
# # check_dEV!(evs, tmpvars, θt, σv, prim, extrema(ψspace), 0.2)
# # check_dEV(θt, σv, prim, extrema(ψspace), 0.2)
# check_dEVσ(evs, tmpvars, θt, σv, prim, extrema(ψspace), 0.2)
#
#
# T = eltype(θt)
# p = prim
# tmp = tmpvars
# roy = 0.2
#
# EV1 = zeros(T, size(evs.EV))
# EV2 = zeros(T, size(evs.EV))
#
# h = max( abs(σv), one(T) ) * cbrt(eps(T))
# σp = σv + h
# σm = σv - h
# hh = σp - σm
#
# update_payoffs!(tmp, θt, σv, p, roy, false; h=-h)
# solve_vf_all!(EV1, tmp.uin, tmp.uex, tmp.ubVfull, tmp.lse, tmp.tmp, tmp.IminusTEVp, p.wp, p.Πz, tmp.βΠψ, p.β)
#
# update_payoffs!(tmp, θt, σv, p, roy, false; h=h)
# solve_vf_all!(EV2, tmp.uin, tmp.uex, tmp.ubVfull, tmp.lse, tmp.tmp, tmp.IminusTEVp, p.wp, p.Πz, tmp.βΠψ, p.β)
#
# solve_vf_all!(evs, tmp, θt, σv, p, roy, true)
#
# dEVk = @view(evs.dEV_σ[:,:,1,1:end-1])
# EV1vw = @view(EV1[:,:,1:size(dEVk,3)])
# EV2vw = @view(EV2[:,:,1:size(dEVk,3)])
# EVfd = (EV2vw .- EV1vw) ./ hh
# EVfd ≈ dEVk  ||  warn("Bad grad for σ at vspace[1]")
#
# absd = maximum( absdiff.(EVfd, dEVk ) )
# reld = maximum( reldiff.(EVfd, dEVk ) )
# println("For σ, abs diff is $absd. max rel diff is $reld")
#
#
# dEVk
#
# tmp.duexσ[:,3,2:4,2:3]
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# #
