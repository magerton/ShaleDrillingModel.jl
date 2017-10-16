using ShaleDrillingModel
using Base.Test
using JLD

pth = joinpath(Pkg.dir("DynamicDrillingProblemEstimation"), "data/price-cost-transitions.jld")
@load pth pspace cspace Πpcr Πp1



β = (1.02 / 1.125) ^ (1./12.)  # real discount rate
royalty_rates = [0.125, 1./6., 0.1875, 0.20, 0.225, 0.25]
geology_types = 1:10
royalty_types = 1:length(royalty_rates)



nψ, dmx, nz, nv =  5, 2, size(Πp1,1), 5
θt, σv = [1.0, -2.6, -2.4, 1.2], 0.5

wp = well_problem(dmx,4,10)
zspace, ψspace, dspace, d1space, vspace = (pspace,), linspace(-5.5, 5.5, nψ), 0:dmx, 0:1, linspace(-3.0, 3.0, nv)
nd = length(dspace)
ns = length(wp)

uin = Array{Float64}(nz,nψ,nd,2)
uex = Array{Float64}(nz,nψ,nd)
duin = Array{Float64}(nz,nψ,length(θt),nd,2)
duex = Array{Float64}(nz,nψ,length(θt),nd)
duexσ = Array{Float64}(nz,nψ,nv,nd)

uin0 = @view(uin[:,:,:,1])
uin1 = @view(uin[:,:,:,2])
duin0 = @view(duin[:,:,:,:,1])
duin1 = @view(duin[:,:,:,:,2])

EV = zeros(Float64, nz,nψ,ns)

i = 17
logsumubV = Array{Float64}(nz,nψ)
ubVmax = Array{Float64}(nz,nψ)
ubVfull = Array{Float64}(nz,nψ,nd)
qtmp = similar(ubVfull)
q0 = @view(qtmp[:,:,1])
EVtmp = Array{Float64}(nz,nψ)
dmaxp1 = dmax(wp,i)+1
ubV = @view(ubVfull[:,:,1:dmaxp1])
Πz = Πp1
EV0 = @view(EV[:,:,i])
IminusTEVp = ShaleDrillingModel.ensure_diagonal(Πz)

# make u
fillflows(uin0, uin1, uex, θt, σv, uflow, makepdct(zspace, ψspace, vspace, wp, θt, :u), 0.2)
fillflows(duin0, duin1, duex, θt, σv, duflow, makepdct(zspace, ψspace, vspace, wp, θt, :du), 0.2)
check_flowgrad(θt, σv, zspace, ψspace, vspace, wp)




# size(duin0)
# size(makepdct(zspace, ψspace, vspace, wp, θt, :du))
#
# size(makepdct(zspace, ψspace, vspace, wp, θt, :u))
# size(uin1)
#
# makeu(uin, uex, θt, zspace, ψspace, dspace, d1space, 0.20, 1)
#
# # try a single VFI
# EV0 .= 0.
# EVtmp .= 0.
# ubV .= @view(uin[:,:,1:dmaxp1,2])
# q    = @view(qtmp[:,:,1:dmaxp1])
# ubV .= @view(uin[:,:,1:dmaxp1,2])
#
# !(0. ≈ 1.)
#
# # ---------------- logsumexp ------------------
#
# # test logsumexp
# let t = similar(logsumubV), lsetest = similar(logsumubV)
#
#     logsumexp3!(logsumubV,ubVmax,ubV)
#     for i in 1:31, j in 1:5
#         lsetest[i,j] = logsumexp(@view(ubV[i,j,:]))
#     end
#     @test all(lsetest .== logsumubV)
#
#     # logsumexp_and_softmax3
#     logsumexp_and_softmax3!(logsumubV, q, ubVmax, ubV)
#     @test all(sum(q,3) .≈ 1.0)
#     @test all(lsetest .== logsumubV)
#
#     # logsumexp_and_softmax3 - q0
#
#     logsumexp_and_softmax3!(logsumubV, t, ubVmax, ubV)
#     @test all(t .== q[:,:,1])
#     @test all(lsetest .== logsumubV)
# end
#
# # ---------------- logsumexp ------------------
#
# # try a single VFI
# vfit!(EVtmp, logsumubV, ubVmax, ubV, Πz)
# extrema(EVtmp .- EV0) .* β ./ (1.0-β)
#
# # try VFI for inf horizon
# let EVvfit = similar(EVtmp),
#     EVpfit = similar(EVtmp)
#
#     ubV .= @view(uin[:,:,1:dmaxp1,2])
#     @show solve_inf_vfit!(EVvfit, EVtmp, logsumubV, ubV, Πz, β, maxit=5000, vftol=1e-11)
#
#     # try a PFI
#     ubV .= @view(uin[:,:,1:dmaxp1,2])
#     pfit!(EV0, EVtmp, logsumubV, q0, ubV, IminusTEVp, Πz, β)
#     @show solve_inf_vfit!(EVpfit, EVtmp, logsumubV, ubV, Πz, β, maxit=12, vftol=1.0)
#     @show solve_inf_pfit!(EVpfit, EVtmp, logsumubV, q0, ubV, IminusTEVp, Πz, β; maxit=20, vftol=1e-11)
#     @test EVpfit ≈ EVvfit
# end
#
# let idxs = [explore_state_inds(wp)..., infill_state_inds(wp)..., terminal_state_ind(wp)...]
#     @test idxs ⊆ 1:length(wp)
#     @test 1:length(wp) ⊆ idxs
# end
#
# solve_vf_terminal!(EV)
# solve_vf_infill!(EV, uin, wp, EVtmp, logsumubV, qtmp, ubVfull, IminusTEVp, Πz, β; maxit0=12, maxit1=20, vftol=1e-11)
#
#
# x = reshape(collect(1:2*3*4), 2,3,4)
#
# permutedims(x, (3,1,2))
# x
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
