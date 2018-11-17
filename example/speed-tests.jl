using ShaleDrillingModel
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
gr()


# -----------------------------------------------------
#    Other VF parameters
# -----------------------------------------------------


# royalty rates
royalty_rates = [0.125, 1.0/6.0, 0.1875, 0.20, 0.225, 0.25]
royalty_types = 1:length(royalty_rates)

# geology types
geology_types = 1.3430409262656042:0.1925954901417719:5.194950729101042

# grids for prices & unobserved heterogeneity
nψ = 41
ψspace = range(-4.0, stop=4.0, length=nψ)

# initial parameters
flowfuncname = :exproy_extend
θt = [3.7, -13.0, 1.8, 2.4, 2.3, -6.5, -4.9, 2.4, -0.2, -1.7,]
σv = 1.55
θfull = vcat(θt, σv)


# discount
β = (1.0234188 / 1.125) ^ (1.0/12.)

# problem sizes
wp = well_problem(7,8,120,60,24)

# load datasets
println("loading transitions")
datadir = joinpath(ENV["JULIA_PKG_DEVDIR"], "ShaleDrillingData", "data")
# pvdat = load(joinpath(datadir, "price-vol-transitions.jld"))  # logpspace logσspace Πp
pdat  = load(joinpath(datadir, "price-transitions.jld"))      # pspace Πp Πp1


# -----------------------------------------------------
#    Transition matrix construction
# -----------------------------------------------------

# make deviations
zrandwalk(x::Real, st::Real, σ::Real) = (x - st) / σ

# range for i,j block in matrix where block is n x n
blockrange(i,j,n) = ((i-1)*n+1):(i*n),  ((j-1)*n+1):(j*n)

extrema_logp = [0.8852633,  2.485073]
extrema_logσ = [-3.372397, -2.060741]

sdlogσ = 0.09381059
nlogp = 45
nlogσ = 13

# sparse versions
minp = 1e-5

# make range for coefs
logσspace = range(extrema_logσ[1] - 3*sdlogσ, stop=extrema_logσ[2] + 3*sdlogσ, length=nlogσ)
logpspace = range(extrema_logp[1] - log(2.0), stop=extrema_logp[2] + log(2.0), length=nlogp)

# create log σ discretization
Pσ, JN, Λ, L_p1, approxErr = discreteNormalApprox(logσspace, logσspace, (x::Real,st::Real) -> zrandwalk(x,st,sdlogσ), 7)
all(sum(Pσ,dims=2) .≈ 1.0) || throw(error("each row of π must sum to 1"))

# match moments for logp
p_from_sigma(logσ::Real) = Dict( [:P, :JN, :Λ, :L, :approxErr,] .=> discreteNormalApprox(logpspace, logpspace, (x::Real,st::Real) -> zrandwalk(x, st, exp(logσ)), 11) )
ps_from_sigma = Dict(logσspace .=> [p_from_sigma(logσ) for logσ in logσspace])

# Fill giant transition matrix
P = Matrix{eltype(logpspace)}(undef, nlogp*nlogσ, nlogp*nlogσ)
for (j, logσ) in enumerate(logσspace)
  for i in 1:nlogσ
    P[blockrange(i,j,nlogp)...] .= Pσ[i, j] .* ps_from_sigma[logσ][:P]
  end
end

# sparsify transigion
Πpvol = MarkovTransitionMatrices.sparsify!(P, minp)
println("Πpvol has $(length(Πpvol.nzval)) values")

# k = 11
# Pxform = reshape(P, nlogp, nlogσ, nlogp, nlogσ)
heatmap(P)
# heatmap(P[nlogp*k+1:nlogp*(k+1), nlogp*k+1:nlogp*(k+1)])
# heatmap(P[11:35:(35*nlogσ), 11:35:(35*nlogσ)])
# heatmap(Pxform[35,:,35,:])
# dropdims(sum(Pxform, dims=(1,3)), dims=(1,3))./35
# sum(Pxform[35,:,35,:],dims=2)

# plot moments matched for log P
plot(logpspace, [ps_from_sigma[sig][:L] for sig in logσspace ], yticks = 1:11, labels = ["$(round(exp(sig); digits=2))" for sig in logσspace], xlabel="Log p", ylabel="Moments matched")
# savefig("D:/projects/royalty-rates-and-drilling/plots/moments-matched-logponly.pdf")


# -----------------------------------------------------
#    speed tests
# -----------------------------------------------------

function prim_tmp_evs(zspace,Πp)
    prim = dcdp_primitives(flowfuncname, β, wp, zspace, Πp, ψspace)
    tmpv = dcdp_tmpvars(prim)
    evs = dcdp_Emax(prim)
    ShaleDrillingModel.check_size(prim, evs)
    zero!(evs)
    return evs, tmpv, prim
end

dograd = false
# pids = [1,]
typidx = (11,3,)
itypes = (geology_types,royalty_rates)
typs = getindex.(itypes, typidx)




println("\n\ndoing NO vol")
tup = prim_tmp_evs((pdat["pspace"],),       pdat["Πp1"])
println("Πp has $(length(tup[3].Πz.nzval)) values")
@btime solve_vf_all!(tup..., θfull, typs, dograd; maxit0=30, maxit1=20, vftol=1e-9)

println("\n\ndoing hi/lo vol")
tup = prim_tmp_evs((pdat["pspace"],1:2,),   pdat["Πp"])
println("Πp has $(length(tup[3].Πz.nzval)) values")
@btime solve_vf_all!(tup..., θfull, typs, dograd; maxit0=30, maxit1=20, vftol=1e-9)

println("\n\ndoing giant price/vol matrix")
tup = prim_tmp_evs((logpspace, logσspace,), Πpvol)
println("Πp has $(length(tup[3].Πz.nzval)) values")
zero!(tup[1])
@btime solve_vf_all!(tup..., θfull, typs, dograd; maxit0=30, maxit1=20, vftol=1e-9)
