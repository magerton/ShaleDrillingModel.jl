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
nψ, dmx, nz, nv, ngeo =  51, 2, size(Πp1,1), 51, 1
wp = well_problem(dmx,4,10)
zspace, ψspace, dspace, d1space, vspace = (pspace,), linspace(-6.0, 6.0, nψ), 0:dmx, 0:1, linspace(-3.0, 3.0, nv)
# nd, ns, nθ = length(dspace), length(wp), length(θt)

prim = dcdp_primitives(u_add, udθ_add, udσ_add, udψ_add, β, wp, zspace, Πp1, ψspace, ngeo, length(θt))
tmpv = dcdp_tmpvars(prim)
evs = dcdp_Emax(prim)

# check sizes of models
ShaleDrillingModel.check_size(prim, evs)


println("testing flow gradients")
@test check_flowgrad(θt, σv, prim, 0.2)
println("testing transition derivatives")
@test check_dΠψ(σv, ψspace)

# include("test_utility.jl")
# include("test_transition.jl")

println("filling per-period payoffs")
fillflows_grad!(tmpv, prim, θt, σv, 0.2)
# include("logsumexp3.jl")
# include("vf_solve_terminal_and_infill.jl")
# include("vf_solve_exploratory.jl")

zero!(tmpv)
solve_vf_all!(evs, tmpv, prim, θt, σv, 0.2, Val{true})

# include("test_dpsi.jl")
# include("parallel_solution.jl")


# ------------------------------- action ----------------------------------

# rmprocs(workers())
# pids = addprocs()
# @everywhere @show pwd()
# @everywhere using ShaleDrillingModel

pids = [1,]
sev = SharedEV(pids, prim, royalty_rates, 1:1)
# @eval @everywhere begin
#     set_g_dcdp_primitives($prim)
#     set_g_dcdp_tmpvars($tmpv)
#     set_g_SharedEV($sev)
# end

isev = ItpSharedEV(sev, prim, σv)
θfull = vcat(θt,σv)
T = eltype(θfull)
tmp = Vector{T}(dmax(wp)+1)
θ1 = similar(θfull)
θ2 = similar(θfull)

println("Testing logP & gradient")


dograd = true

rngs = (zspace..., vspace, vspace, 1:dmax(wp)+1, 1:length(wp), royalty_rates, 1:1)
idxs = (1:2:31, 1:5:51, 1:5:51, 1:dmax(wp)+1, 1:length(wp), royalty_rates, 1:1,)
grad = zeros(T, length.((θfull, idxs...)))
fdgrad = zeros(T, size(grad))
CR = CartesianRange(length.(idxs))
grad .= zero(T)
fdgrad .= zero(T)


θttmp = Vector{Float64}(length(θfull) - prim.ngeo)
@code_warntype logP!(Vector{Float64}(5), tmp, θttmp, θfull, prim, isev, (1.,1.), (1.2,), 1, 1, (1,1,), true)

# parallel_solve_vf_all!(sev, θfull, Val{dograd})
# println("solved round 1. doing logP")
# i = 1
# for CI in CR
#     zi, ui, vi, di, si, ri, gi = CI.I
#     z, u, v, d, s, r, g = getindex.(rngs, CI.I)
#     if i == 1
#         @show CI.I
#         @show getindex.(rngs, CI.I)
#     end
#     i+= 1
#     if d <= dmax(wp,s)+1
#         @views lp = logP!(grad[:,CI], tmp, θfull, prim, isev,  (u,v), (z,), d, (ri, gi,), dograd, true)
#     end
# end
#
# dograd = false
# for k in 1:length(θfull)
#     println("θfull[$k]...")
#     θ1 .= θfull
#     θ2 .= θfull
#     h = peturb(θfull[k])
#     θ1[k] -= h
#     θ2[k] += h
#     hh = θ2[k] - θ1[k]
#
#     parallel_solve_vf_all!(sev, θ1, Val{dograd})
#     println("logp: θ[$k]-h")
#     for CI in CR
#         zi, ui, vi, di, si, ri, gi = CI.I
#         z, u, v, d, s, r, g = getindex.(rngs, CI.I)
#         if d <= dmax(wp,s)+1
#             fdgrad[k,CI] -= logP!(Vector{T}(0), tmp, θ1, prim, isev, dograd, (ri,gi), (u,v), d, s, z...)
#         end
#     end
#
#     println("solving again for logp: θ[$k]+h")
#     parallel_solve_vf_all!(sev, θ2, Val{dograd})
#     println("updating fdgrad")
#     for CI in CR
#         zi, ui, vi, di, si, ri, gi = CI.I
#         z, u, v, d, s, r, g = getindex.(rngs, CI.I)
#         if d <= dmax(wp,s)+1
#             fdgrad[k,CI] += logP!(Vector{T}(0), tmp, θ2, prim, isev, dograd, (ri,gi), (u,v), d, s, z...)
#             fdgrad[k,CI] /= hh
#         end
#     end
# end
# println("checking gradient")
#
# @views maxv, idx =  findmax(abs.(grad[1:end-1,:,:,:,:,:,:,:] .- fdgrad[1:end-1,:,:,:,:,:,:,:]))
# println("worst value is $maxv at $sub for dlogP WITHOUT σ.")
# @test maxv < 1e-7
#
# println("With σ")
# maxv, idx =  findmax(abs.(grad[end,:,:,:,:,:,:,:] .- fdgrad[end,:,:,:,:,:,:,:]))
# @views mae = mean(abs.(grad[end,:,:,:,:,:,:,:] .- fdgrad[end,:,:,:,:,:,:,:]))
# @views mse = var(grad[end,:,:,:,:,:,:,:] .- fdgrad[end,:,:,:,:,:,:,:])
# @views med = median(abs.(grad[end,:,:,:,:,:,:,:] .- fdgrad[end,:,:,:,:,:,:,:]))
# @views q90 = quantile(vec(abs.(grad[end,:,:,:,:,:,:,:] .- fdgrad[end,:,:,:,:,:,:,:])), 0.9)
# @views sub = ind2sub(grad[end,:,:,:,:,:,:,:], idx)
# vals = getindex.(rngs, sub)
# println("worst value is $maxv at $sub for dlogP. This has characteristics $vals")
# println("MAE = $mae. MSE = $mse. Median abs error = $med. 90pctile = $q90")
#
#
#
#
# # @test 0.0 < maxv < 1.5e-3
# # @test maxv < maxv_itp
# # @test isapprox(grad, fdgrad, atol=1e-7)
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















# include("pre-action-probabilities.jl")
# include("action_probabilities.jl")


#
