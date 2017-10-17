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
dEV_σ = zeros(Float64,nz,nψ,nv,length(explore_state_inds(wp)))

# set up temp variables
logsumubV = Array{Float64}(nz,nψ)
ubVmax    = Array{Float64}(nz,nψ)
EVtmp     = Array{Float64}(nz,nψ)
ubVfull = Array{Float64}(nz,nψ,nd)
qtmp    = Array{Float64}(nz,nψ,nd)
q0  = @view(qtmp[:,:,1])

dubVfull   = Array{Float64}(nz,nψ,nθ,nd)
sumdubV    = Array{Float64}(nz,nψ,nθ)
Πz_sumdubV = Array{Float64}(nz,nψ,nθ)


Πz = Πp1
IminusTEVp = ShaleDrillingModel.ensure_diagonal(Πz)

# ----------------- test flow payoffs ----------------------

# make u
fillflows(uin0, uin1, uex, θt, σv, uflow, makepdct(zspace, ψspace, vspace, wp, θt, :u), 0.2)
fillflows(duin0, duin1, duex, θt, σv, duflow, makepdct(zspace, ψspace, vspace, wp, θt, :du), 0.2)
check_flowgrad(θt, σv, zspace, ψspace, vspace, wp)

@test all(uin0[:,:,1] .== 0.)
@test all(uin1[:,:,1] .== 0.)
@test all(uex[:,:,1] .== 0.)

# ------------------- setup views --------------------

i = 17
dmaxp1 = dmax(wp,i)+1

# set up action-space specific views
EV0  = @view(EV[:,:,i])
dEV0 = @view(dEV[:,:,:,i])
ubV  = @view(ubVfull[:,:,1:dmaxp1])
dubV = @view(dubVfull[:,:,:,1:dmaxp1])
q    = @view(qtmp[:,:,1:dmaxp1])


EV0 .= 0.0
EVtmp .= 0.0
dEV0 .= 0.0
ubV .= @view(uin[:,:,1:dmaxp1,2])


# ---------------- logsumexp ------------------

# test logsumexp
let t = similar(logsumubV), lsetest = similar(logsumubV)

    logsumexp3!(logsumubV,ubVmax,ubV)
    for i in 1:31, j in 1:5
        lsetest[i,j] = logsumexp(@view(ubV[i,j,:]))
    end
    @test all(lsetest .== logsumubV)

    # logsumexp_and_softmax3
    logsumexp_and_softmax3!(logsumubV, q, ubVmax, ubV)
    @test all(sum(q,3) .≈ 1.0)
    @test all(lsetest .== logsumubV)

    # logsumexp_and_softmax3 - q0
    logsumexp_and_softmax3!(logsumubV, t, ubVmax, ubV)
    @test all(t .== q[:,:,1])
    @test all(lsetest .== logsumubV)
end

# ---------------- Regime 2 VFI and PFI ------------------

# try a single VFI
ShaleDrillingModel.vfit!(EVtmp, logsumubV, ubVmax, ubV, Πz)
extrema(EVtmp .- EV0) .* β ./ (1.0-β)

# try VFI for inf horizon
if true
    let EVvfit = zeros(eltype(EVtmp), size(EVtmp)),
        EVpfit = zeros(eltype(EVtmp), size(EVtmp)),
        dEVvfit = zeros(eltype(dEV0), size(dEV0)),
        dEVpfit = zeros(eltype(dEV0), size(dEV0))

        # solve with VFI only
        EV .= 0.0
        ubV .= @view(uin0[:,:,1:dmaxp1])
        @show ShaleDrillingModel.solve_inf_vfit!(EVvfit, EVtmp, logsumubV, ubV, Πz, β, maxit=5000, vftol=1e-12)

        # solve with hybrid iteration (12 VFit steps + more PFit)
        ubV .= @view(uin0[:,:,1:dmaxp1])
        ShaleDrillingModel.pfit!(EV0, EVtmp, logsumubV, q0, ubV, IminusTEVp, Πz, β)
        @show ShaleDrillingModel.solve_inf_vfit!(EVpfit, EVtmp, logsumubV, ubV, Πz, β, maxit=12, vftol=1.0)
        @show ShaleDrillingModel.solve_inf_pfit!(EVpfit, EVtmp, logsumubV, q0, ubV, IminusTEVp, Πz, β; maxit=20, vftol=1e-11)
        @test EVpfit ≈ EVvfit

        # update ubV and make inf horizon derivatives
        ubV .= @view(uin0[:,:,1:dmaxp1])
        ubV[:,:,1] .+=  β .* EVpfit
        dubV .= @view(duin0[:,:,:,1:dmaxp1])
        ShaleDrillingModel.gradinf!(dEVpfit, ubV, dubV, logsumubV, ubVmax, sumdubV, Πz_sumdubV, IminusTEVp, Πz, β)  # note: destroys ubV

        # update dubV with gradinf! results & test that when we run VFI grad, we get the same thing back.
        # Note: since ubV destroyed, re-make
        dubV[:,:,:,1] .+= β .* dEVpfit
        ubV .= @view(uin0[:,:,1:dmaxp1])
        ubV[:,:,1] .+= β .* EVpfit
        ShaleDrillingModel.vfit!(EVvfit, dEVvfit, logsumubV, ubVmax, q, sumdubV, ubV, dubV, Πz)
        @test EVpfit ≈ EVvfit
        @test dEVpfit ≈ dEVvfit
    end
end


# if true
#
#     let idxs = [ShaleDrillingModel.explore_state_inds(wp)..., ShaleDrillingModel.infill_state_inds(wp)..., ShaleDrillingModel.terminal_state_ind(wp)...]
#         @test idxs ⊆ 1:length(wp)
#         @test 1:length(wp) ⊆ idxs
#     end
#
#     # full VFI
#     EV .= 0.0
#     ShaleDrillingModel.solve_vf_terminal!(EV)
#     ShaleDrillingModel.solve_vf_infill!(EV, uin, wp, EVtmp, logsumubV, qtmp, ubVfull, IminusTEVp, Πz, β; maxit0=12, maxit1=20, vftol=1e-11)
#     @test !all(EV .== 0.)
#
# end
#
# # ---------------- Regime 1 VFI and PFI ------------------
#
# if true
#
#     βΠψ = Matrix{eltype(σv)}(nψ,nψ)
#     βdΠψ = similar(βΠψ)
#     tauchen86_σ!(βΠψ,βdΠψ,ψspace,σv)
#     βΠψ .*= β
#     βdΠψ .*= β
#
#     ShaleDrillingModel.solve_vf_explore!(EV, uex, wp, EVtmp, logsumubV, qtmp, ubVfull, Πz, βΠψ, β)
#     @test !all(EV[:,:,explore_state_inds(wp)] .== 0.)
#
#
#     using Plots
#     gr()
#     plot(exp.(pspace), EV[:,3,explore_state_inds(wp)])
#
# end
#
#



#
