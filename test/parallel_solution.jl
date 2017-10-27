println("testing parallel solution")

rmprocs(workers())
pids = addprocs()
@everywhere @show pwd()
@everywhere using ShaleDrillingModel

sev = SharedEV(pids, prim, royalty_rates, 1:1)
@eval @everywhere begin
    set_g_dcdp_primitives($prim)
    set_g_dcdp_tmpvars($tmpv)
    set_g_SharedEV($sev)
end
s = remotecall_fetch(get_g_SharedEV, pids[2])
fetch(s)

zero!(sev)
@show parallel_solve_vf_all!(sev, vcat(θt,σv), Val{true})
@test all(isfinite.(sev.EV))
@test all(isfinite.(sev.dEV))
@test all(isfinite.(sev.dEVσ))
@test all(isfinite.(sev.dEVψ))
@test !all(sev.EV .== 0.0)
@test !all(sev.dEV .== 0.0)
@test !all(sev.dEVσ .== 0.0)
@test !all(sev.dEVψ .== 0.0)

rmprocs(workers())



# rmprocs(workers())
# pids = addprocs()
# @everywhere @show pwd()
# @everywhere using ShaleDrillingModel
# @eval @everywhere begin
#     set_g_dcdp_primitives($prim)
#     set_g_dcdp_tmpvars($tmpv)
#     set_g_SharedEV($sev)
# end
# parallel_solve_vf_all!(sev, θfull, Val{dograd})
# parallel_solve_vf_all!(sev, θfull, Val{false}; maxit0=12, maxit1=20, vftol=1e-10)
