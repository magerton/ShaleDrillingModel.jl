using ShaleDrillingModel
using ShaleDrillingData
using Test
using StatsFuns
using FileIO
using Interpolations
using Statistics
using SparseArrays
using BenchmarkTools
using MarkovTransitionMatrices
using Plots
using SparseArrays


# -----------------------------------------------------
#    Other VF parameters
# -----------------------------------------------------


# royalty rates
royalty_rates = [0.125, 1.0/6.0, 0.1875, 0.20, 0.225, 0.25]
royalty_types = 1:length(royalty_rates)

# geology types
geology_types = range(1.3430409262656042, stop=5.194950729101042, length=21)

# grids for prices & unobserved heterogeneity
nψ = 41
ψspace = range(-4.0, stop=4.0, length=nψ)

# initial parameters
flowfuncname = :dgt1_cost_restr
θt = [-4.623, -6.309, -4.957, -0.2, -0.47,]
σv = 0.1
θfull = vcat(θt, σv)

# discount
β = (1.0234188 / 1.125) ^ (1.0/4.)

# problem sizes
wp = well_problem(7,8,40,20,8)

# sparse versions
minp = 1e-4

# make deviations
zrandwalk(x1::Real, mu::Real, σ::Real) = (x1 - mu) / σ
Elogc(x0::Real) = x0 + 0.1358467453 - 0.0007906602 * exp(x0)

nlogp = 25
nlogc = 13
nlogσ = 9

extrema_logp = [0.8510954,  2.405270]
extrema_logc = [5.0805023,  5.318393]
extrema_logσ = [-2.9558226, -1.542508]
sdlogσ = 0.18667531
cσ_to_pσ = 0.02777901 / 0.08488665

    # make range for coefs
logσ_space = range(extrema_logσ[1] - 2*sdlogσ, stop=extrema_logσ[2] + 2*sdlogσ, length=nlogσ)
logp_space = range(extrema_logp[1] - log(2.0), stop=extrema_logp[2] + log(2.0), length=nlogp)
logc_space = range(extrema_logc[1] - log(2.0), stop=extrema_logc[2] + log(2.0), length=nlogc)

# range for i,j block in matrix where block is n x n
blockrange(i,j,n) = ((i-1)*n+1):(i*n),  ((j-1)*n+1):(j*n)

# create log σ discretization
println("Discretizing log σ")
Pσ, JN, Λ, L_p1, approxErr = discreteNormalApprox(logσ_space, logσ_space, (x::Real,st::Real) -> zrandwalk(x,st,sdlogσ), 7)
all(sum(Pσ,dims=2) .≈ 1.0) || throw(error("each row of π must sum to 1"))

# match moments for logp
p_from_sigma(logσ::Real) = Dict( [:P, :JN, :Λ, :L, :approxErr,] .=> discreteNormalApprox(logp_space, logp_space, (x1::Real,x0::Real) -> zrandwalk(x1, x0, exp(logσ)), 11) )
c_from_sigma(logσ::Real) = Dict( [:P, :JN, :Λ, :L, :approxErr,] .=> discreteNormalApprox(logc_space, logc_space, (x1::Real,x0::Real) -> zrandwalk(x1, x0, exp(logσ)*cσ_to_pσ), 5) )

println("Discretizing log p and log c")
ps_from_sigma = [p_from_sigma(logσ) for logσ in logσ_space]
cs_from_sigma = [c_from_sigma(logσ) for logσ in logσ_space]

# Fill giant transition matrix
println("Filling transition matrix")
P  = Matrix{eltype(logp_space)}(undef, nlogp*nlogσ, nlogp*nlogσ)
PC = Matrix{eltype(logp_space)}(undef, nlogp*nlogc*nlogσ, nlogp*nlogc*nlogσ)
for (j, logσj) in enumerate(logσ_space)
  for (i, logσi) in enumerate(logσ_space)
    P[blockrange(i,j,nlogp)...]        .= Pσ[i, j] .* ps_from_sigma[i][:P]
    PC[blockrange(i,j,nlogp*nlogc)...] .= Pσ[i, j] .* kron( cs_from_sigma[i][:P],  ps_from_sigma[i][:P])
  end
end

# sparsify transigion
Πp = MarkovTransitionMatrices.sparsify!(P, minp)
Πpc = MarkovTransitionMatrices.sparsify!(PC, minp)
P = 0.0 # destroy P
PC = 0.0
zspace = (logp_space, logσ_space,) # (logp_space, logc_space, logσ_space,)
zcspace = (logp_space, logc_space, logσ_space,)

# sparsify transigion
println("Πp has $(length(Πp.nzval)) values")
println("Πpc has $(length(Πpc.nzval)) values")

# k = 11
# Pxform = reshape(P, nlogp, nlogσ, nlogp, nlogσ)
# heatmap(P)
# heatmap(P[nlogp*k+1:nlogp*(k+1), nlogp*k+1:nlogp*(k+1)])
# heatmap(P[11:35:(35*nlogσ), 11:35:(35*nlogσ)])
# heatmap(Pxform[35,:,35,:])
# dropdims(sum(Pxform, dims=(1,3)), dims=(1,3))./35
# sum(Pxform[35,:,35,:],dims=2)

# plot moments matched for log P
# plot(logp_space, [ps_from_sigma[sig][:L] for sig in logσ_space ], yticks = 1:11, labels = ["$(round(exp(sig); digits=2))" for sig in logσ_space], xlabel="Log p", ylabel="Moments matched")
# savefig("D:/projects/royalty-rates-and-drilling/plots/moments-matched-logponly.pdf")


# -----------------------------------------------------
#    speed tests
# -----------------------------------------------------

function prim_tmp_evs(FFname, zzspace, bigp)
    prim = dcdp_primitives(FFname, β, wp, zzspace, bigp, ψspace)
    tmpv = dcdp_tmpvars(prim)
    evs = dcdp_Emax(prim)
    ShaleDrillingModel.check_size(prim, evs)
    zero!(evs)
    return evs, tmpv, prim
end

dograd = false
pids = [1,]
typidx = (11,3,)
itypes = (geology_types,royalty_rates,)
typs = getindex.(itypes, typidx)

# println("\n\ndoing NO vol")
# tup = prim_tmp_evs((pdat["pspace"],),       pdat["Πp1"])
# println("Πp has $(length(tup[3].Πz.nzval)) values")
# @btime solve_vf_all!(tup..., θfull, typs, dograd; maxit0=30, maxit1=20, vftol=1e-9)
#
# println("\n\ndoing hi/lo vol")
# tup = prim_tmp_evs((pdat["pspace"],1:2,),   pdat["Πp"])
# println("Πp has $(length(tup[3].Πz.nzval)) values")
# @btime solve_vf_all!(tup..., θfull, typs, dograd; maxit0=30, maxit1=20, vftol=1e-9)

println("\n\ndoing giant price/vol matrix")
tup = prim_tmp_evs(:dgt1_cost_restr, zcspace, Πpc)
println("Πz has $(length(tup[3].Πz.nzval)) values")
zero!(tup[1])

@show @btime solve_vf_all!(tup..., θfull, typs, dograd; maxit0=30, maxit1=20, vftol=1e-8)
@show @btime solve_vf_all!(tup..., θfull, typs, dograd; maxit0=40, maxit1=20, vftol=1e-8)
@show @btime solve_vf_all!(tup..., θfull, typs, dograd; maxit0=50, maxit1=20, vftol=1e-8)

tup = prim_tmp_evs(:dgt1_restr, zspace, Πp)
println("Πz has $(length(tup[3].Πz.nzval)) values")
zero!(tup[1])
@btime solve_vf_all!(tup..., θfull, typs, dograd; maxit0=40, maxit1=20, vftol=1e-9)



# println("\nSolving for all types")
# sev = SharedEV(pids, tup[3], geology_types[1:3], royalty_rates[1:2])
# @btime serial_solve_vf_all!(sev, tup[2], tup[3], θfull, Val{dograd}; maxit0=30, maxit1=20, vftol=1e-9)
