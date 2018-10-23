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
    set_g_ItpSharedEV(ItpSharedEV(get_g_SharedEV(), get_g_dcdp_primitives(), $σv))
end
s = remotecall_fetch(get_g_SharedEV, pids[2])
fetch(s)

zero!(sev)
@show parallel_solve_vf_all!(sev, vcat(θt,σv), Val{true})
@test all(isfinite.(sev.EV))
@test all(isfinite.(sev.dEV))
@test all(isfinite.(sev.dEVσ))
@test !all(sev.EV .== 0.0)
@test !all(sev.dEV .== 0.0)
@test !all(sev.dEVσ .== 0.0)



let EVcopy = similar(sev.EV),
    EVcoefs = similar(sev.EV),
    sitev = get_g_ItpSharedEV()

    @test(sitev.EV.itp.coefs === sev.EV)
    # store EV
    EVcopy .= sev.EV
    #interpolate & store
    serial_prefilterByView!(sev,sitev,true)
    EVcoefs .= sev.EV
    # check interpolation happened
    @test !all(EVcopy .== EVcoefs)
    # restore coefs
    sev.EV .= EVcopy
    @test all(sitev.EV.itp.coefs .== EVcopy)
    parallel_prefilterByView!(sev,sitev,true)
    # double check that parallel does the same as serial
    @test !all(EVcopy .== sev.EV)
    @test all(EVcoefs .== sev.EV)
end

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
