# using ShaleDrillingModel
using JLD
using Plots
gr()


include("makeStateSpace.jl")

path_prices = joinpath(Pkg.dir("ShaleDrillingData"), "data", "price-transitions.jld")
@load path_prices pspace Πp

jldpath = joinpath(Pkg.dir("ShaleDrillingData"), "results", "results-full-exp-M2500.jld")
@load jldpath coefmat_full_OPG

# some primitives
β = (1.02 / 1.125) ^ (1./12.)  # real discount rate
royalty_rates = [0.125, 1./6., 0.1875, 0.20, 0.225, 0.25]
geology_types = 1:10
royalty_types = 1:length(royalty_rates)

# initial parameters
θfull = coefmat_full_OPG[2:9,1]
θt = θfull[1:end-1]
σ = θfull[end]

# problem sizes
Πz = Πp
nψ, nz =  51, size(Πz,1)
wp = well_problem(3,4,10)
zspace, ψspace = (pspace,1:2,), range(-4.0, stop=4.0, length=nψ)
prim = dcdp_primitives(:exp, β, wp, zspace, Πp, ψspace)
tmpv = dcdp_tmpvars(prim)
evs = dcdp_Emax(prim)

# check sizes of models
itype = (1./6., 8,)
ShaleDrillingModel.check_size(prim, evs)
solve_vf_all!(evs, tmpv, prim, θt, σ, itype, Val{false})

fillflows!(tmpv, prim, θt, σ, itype...)

ubv = tmpv.uin[:,:,1:2,1] .+ β .* evs.EV[:,:,end-1:end]
q2 = similar(ubv)
softmax3!(q2, tmpv.lse, tmpv.tmp, ubv)


EU = similar(evs.EV)
ubUfull = similar(tmpv.ubVfull)
q = similar(tmpv.ubVfull)
rin = similar(tmpv.uex)
rex = similar(tmpv.uex)
P = zeros(Float64,nz,nψ,dmax(wp)+1,length(wp))
pdct = makepdct(prim, Val{:u})
ShaleDrillingModel.fillflowrevs!(Val{:exp}, ShaleDrillingModel.flowrev, rin, rex, θt, σ, pdct, itype...)

size(P)
size(evs.EV)

prDrill_infill!(
    evs.EV, tmpv.uin, tmpv.uex, tmpv.ubVfull,
    EU, rin, rex, ubUfull,
    P, q,
    tmpv.lse, tmpv.tmp, tmpv.IminusTEVp,
    prim.wp, prim.Πz,
    tmpv.Πψtmp, prim.ψspace, σ, prim.β
    )


tmpc = dcdp_tmpcntrfact(prim)
prDrill_infill!(evs, tmpv, prim, EU, P, tmpc, θfull, (1./6., 8,) )
size(P)

prdrill = 1.0 .- P[:,:,1,:]
plot(ψspace, prdrill[17,:,:])



# let prim = dcdp_primitives(:exp, β, wp, zspace, Πz, ψspace),
#     tmpv = dcdp_tmpvars(prim),
#     sev = SharedEV([1,], prim, [1.0/8.0], 1:1),
#     zdims = length.(prim.zspace),
#     nψ = _nψ(prim),
#     nd = _nd(prim),
#     sEU = Array{Float64}(size(sev.EV)),
#     sP = Array{Float64}(zdims..., nψ, nd, length(prim.wp), length.(sev.itypes)...)
#
#     serial_solve_vf_all!(sev, tmpv, prim, θfull, Val{false})
#     serial_counterfact_all!(sev, tmpv, prim, sEU, sP, tmpc, θfull)
# end




# rmprocs(workers())
# pids = IN_SLURM ? addprocs_slurm(parse(Int, ENV["SLURM_TASKS_PER_NODE"])) : addprocs()

# @everywhere @show pwd()
# @everywhere using ShaleDrillingModel
#
# sev = SharedEV(pids, prim, royalty_rates, 1:1)
# @eval @everywhere begin
#     set_g_dcdp_primitives($prim)
#     set_g_dcdp_tmpvars($tmpv)
#     set_g_SharedEV($sev)
#     set_g_ItpSharedEV(ItpSharedEV(get_g_SharedEV(), get_g_dcdp_primitives(), $σv))
# end
# s = remotecall_fetch(get_g_SharedEV, pids[2])
# fetch(s)
#
# zero!(sev)
# @show parallel_solve_vf_all!(sev, vcat(θt,σv), Val{true})
