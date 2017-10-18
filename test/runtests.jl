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

# ----------------------------------------------------------
# ----------------------------------------------------------
# ----------------------------------------------------------
# ----------------------------------------------------------


# flow utility arrays
uin = Array{Float64}(nz,nψ,nd,2)
uex = Array{Float64}(nz,nψ,nd)
duin = Array{Float64}(nz,nψ,nθ,nd,2)
duex = Array{Float64}(nz,nψ,nθ,nd)
duexσ = Array{Float64}(nz,nψ,nv,nd)

uin0, uin1 = @view(uin[:,:,:,1]), @view(uin[:,:,:,2])
duin0, duin1 = @view(duin[:,:,:,:,1]), @view(duin[:,:,:,:,2])

# value function arrays
EV = zeros(Float64, nz,nψ,ns)
dEV = zeros(Float64, nz,nψ,nθ,ns)
dEV_σ = zeros(Float64,nz,nψ,nv,length(explore_state_inds(wp))+1)

# transition matrices
βΠψ = Matrix{eltype(σv)}(nψ,nψ)
βdΠψ = similar(βΠψ)
Πz = Πp1
IminusTEVp = ShaleDrillingModel.ensure_diagonal(Πz)

# other temp variables
ubVfull  = Array{Float64}(nz,nψ,nd)
dubVfull = Array{Float64}(nz,nψ,nθ,nd)
dubV_σ   = Array{Float64}(nz,nψ,nv,ShaleDrillingModel.exploratory_dmax(wp)+1)
q        = Array{Float64}(nz,nψ,nd)
q0       = @view(q[:,:,1])
tmp      = Array{Float64}(nz,nψ)
lse      = Array{Float64}(nz,nψ)


fa = ShaleDrillingModel.u_add
dfa = ShaleDrillingModel.du_add
dfσa = ShaleDrillingModel.duσ_add

prim = dcdp_primitives(fa, dfa, dfσa, β, wp, zspace, Πp1, nψ, vspace, 1)
tmpvars = dcdp_tmpvars(nθ, prim)
evs = dcdp_Emax(EV,dEV,dEV_σ)

check_size(θt, prim,evs)

update_payoffs!(uin, uex, βΠψ,                          fa,            θt, σv, β, 0.2, zspace, ψspace, vspace, wp)
update_payoffs!(uin, uex, βΠψ, duin, duex, duexσ, βdΠψ, fa, dfa, dfσa, θt, σv, β, 0.2, zspace, ψspace, vspace, wp)
update_payoffs!(tmpvars, θt, σv, prim, extrema(ψspace), 0.2, true)
check_flowgrad(θt, σv, prim, extrema(ψspace), 0.2)

# ----------------- test flow payoffs ----------------------

# make u
fillflows(uin0, uin1, uex,    θt, σv, fa,   makepdct(zspace, ψspace, vspace, wp, θt, :u),   0.2)
fillflows(duin0, duin1, duex, θt, σv, dfa,  makepdct(zspace, ψspace, vspace, wp, θt, :du),  0.2)
fillflows(duexσ,              θt, σv, dfσa, makepdct(zspace, ψspace, vspace, wp, θt, :duσ), 0.2)

check_flowgrad(θt, σv,  fa, dfa, dfσa,    zspace, ψspace, vspace, wp, 0.2)

# are all transitions 0 for no action?
@test all(uin0[:,:,1] .== 0.)
@test all(uin1[:,:,1] .== 0.)
@test all(uex[:,:,1] .== 0.)
@test all(duin0[:,:,:,1] .== 0.)
@test all(duin1[:,:,:,1] .== 0.)
@test all(duex[:,:,:,1] .== 0.)


# ------------------- setup views --------------------

i = 17
dmaxp1 = dmax(wp,i)+1

# set up action-space specific views
EV0  = @view(EV[:,:,i])
dEV0 = @view(dEV[:,:,:,i])
ubV  = @view(ubVfull[:,:,1:dmaxp1])
dubV = @view(dubVfull[:,:,:,1:dmaxp1])

# set values
EV0 .= 0.0
dEV0 .= 0.0
ubV .= @view(uin[:,:,1:dmaxp1,2])
dubV .= @view(duin[:,:,:,1:dmaxp1,2])

# ---------------- logsumexp ------------------

# test logsumexp
let tst = similar(lse),
    lsetest = similar(lse),
    qvw = @view(q[:,:,1:dmaxp1])

    logsumexp3!(lse,tmp,ubV)
    for i in 1:31, j in 1:5
        lsetest[i,j] = logsumexp(@view(ubV[i,j,:]))
    end
    @test all(lsetest .== lse)

    # logsumexp_and_softmax3
    logsumexp_and_softmax3!(lse, qvw, tmp, ubV)
    @test all(sum(qvw,3) .≈ 1.0)
    @test all(lsetest .== lse)

    # logsumexp_and_softmax3 - q0
    logsumexp_and_softmax3!(lse, tst, tmp, ubV)
    @test all(tst .== qvw[:,:,1])
    @test all(lsetest .== lse)
end

# ---------------- Regime 2 VFI and PFI ------------------

# try a single VFI
let EVtmp = zeros(nz,nψ)
    ShaleDrillingModel.vfit!(EVtmp, ubV, lse, tmp, Πz)
    @show extrema(EVtmp .- EV0) .* β ./ (1.0-β)
end

# try VFI for inf horizon
if false
    let EVvfit = zeros(eltype(EV0), size(EV0)),
        EVpfit = zeros(eltype(EV0), size(EV0)),
        dEVvfit = zeros(eltype(dEV0), size(dEV0)),
        dEVpfit = zeros(eltype(dEV0), size(dEV0))

        # solve with VFI only
        EV .= 0.0
        ubV .= @view(uin0[:,:,1:dmaxp1])
        @show ShaleDrillingModel.solve_inf_vfit!(EVvfit, ubV, lse, tmp, Πz, β, maxit=5000, vftol=1e-12)

        # solve with hybrid iteration (12 VFit steps + more PFit)
        ubV .= @view(uin0[:,:,1:dmaxp1])
        ShaleDrillingModel.pfit!(EV0, ubV, lse, tmp, IminusTEVp, Πz, β)
        @show ShaleDrillingModel.solve_inf_vfit!(EVpfit, ubV, lse, tmp,             Πz, β, maxit=12, vftol=1.0)
        @show ShaleDrillingModel.solve_inf_pfit!(EVpfit, ubV, lse, tmp, IminusTEVp, Πz, β; maxit=20, vftol=1e-11)
        @test EVpfit ≈ EVvfit

        # update ubV and make inf horizon derivatives
        ubV .= @view(uin0[:,:,1:dmaxp1])
        ubV[:,:,1] .+=  β .* EVpfit
        dubV .= @view(duin0[:,:,:,1:dmaxp1])
        ShaleDrillingModel.gradinf!(dEVpfit, ubV, dubV, lse, tmp, IminusTEVp, Πz, β)  # note: destroys ubV

        # update dubV with gradinf! results & test that when we run VFI grad, we get the same thing back.
        # Note: since ubV destroyed, re-make
        ubV .= @view(uin0[:,:,1:dmaxp1])
        ubV[:,:,1] .+= β .* EVpfit
        dubV .= @view(duin0[:,:,:,1:dmaxp1])
        dubV[:,:,:,1] .+= β .* dEVpfit
        ShaleDrillingModel.vfit!(EVvfit, dEVvfit, ubV, dubV, lse, tmp, Πz)
        @test EVpfit ≈ EVvfit
        @test dEVpfit ≈ dEVvfit
    end
end


if true
    let idxs = [ShaleDrillingModel.explore_state_inds(wp)..., ShaleDrillingModel.infill_state_inds(wp)..., ShaleDrillingModel.terminal_state_ind(wp)...]
        @test idxs ⊆ 1:length(wp)
        @test 1:length(wp) ⊆ idxs
    end

    # full VFI
    ShaleDrillingModel.solve_vf_terminal!(EV, dEV, dEV_σ)
    ShaleDrillingModel.solve_vf_infill!(EV, uin, ubVfull, lse, tmp, IminusTEVp, wp, Πz, β)
    ShaleDrillingModel.solve_vf_infill!(EV, dEV, uin, duin, ubVfull, dubVfull, lse, tmp, IminusTEVp, wp, Πz, β)
    @test !all(EV .== 0.)
end

# ---------------- Regime 1 VFI and PFI ------------------

if true
    tauchen86_σ!(βΠψ,βdΠψ,ψspace,σv)
    βΠψ .*= β
    βdΠψ .*= β

    ShaleDrillingModel.solve_vf_explore!(EV, uex, ubVfull, lse, tmp, wp, Πz, βΠψ, β)
    @test !all(EV[:,:,explore_state_inds(wp)] .== 0.)

    ShaleDrillingModel.solve_vf_explore!(EV, dEV, dEV_σ, uex, duex, duexσ, ubVfull, dubVfull, dubV_σ, q, lse, tmp, wp, Πz, βΠψ, βdΠψ, β)
end


if true
    ShaleDrillingModel.solve_vf_all!(EV,             uin, uex,                    ubVfull,                      lse, tmp, IminusTEVp, wp, Πz, βΠψ, β)
    ShaleDrillingModel.solve_vf_all!(EV, dEV, dEV_σ, uin, uex, duin, duex, duexσ, ubVfull, dubVfull, dubV_σ, q, lse, tmp, IminusTEVp, wp, Πz, βΠψ, βdΠψ, β)
end


solve_vf_all!(EV,             tmpvars, θt, σv, prim, extrema(ψspace), 0.2,)
solve_vf_all!(EV, dEV, dEV_σ, tmpvars, θt, σv, prim, extrema(ψspace), 0.2, false)
solve_vf_all!(EV, dEV, dEV_σ, tmpvars, θt, σv, prim, extrema(ψspace), 0.2, true)
solve_vf_all!(evs,            tmpvars, θt, σv, prim, extrema(ψspace), 0.2, true)
solve_vf_all!(evs,            tmpvars, θt, σv, prim, extrema(ψspace), 0.2, false)



check_EVgrad(θt, σv, prim, extrema(ψspace), 0.2)





# TODO: make VF structs
# 1) assumptions: primitives: β, wp, Πz, zspace, ψspace, vspace
# 2) parameters: θt, σv, royalty, geology(baked in to θ)
# 3) value function + gradient (must check out)
# 4) tempvars: including flow payoffs & βΠψ (based on assumptions + parameters)













#
