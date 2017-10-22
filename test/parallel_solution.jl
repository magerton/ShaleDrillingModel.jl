println("testing parallel solution")

pids = addprocs()
@everywhere @show pwd()
@everywhere using ShaleDrillingModel

sev = SharedEV(pids, vcat(θt, σv), prim, royalty_rates, 1:1)

@eval @everywhere begin
    set_g_dcdp_primitives($prim)
    set_g_dcdp_tmpvars($tmpv)
    set_g_SharedEV($sev)
end

s = remotecall_fetch(get_g_SharedEV, pids[2])
s = @spawnat pids[2] dcdp_Emax(get_g_SharedEV(),1,1)
fetch(s)

evsvw, typs = dcdp_Emax(sev, 1,1)
solve_vf_all!(evsvw, tmpv, prim, vcat(θt,σv), typs..., Val{true})
solve_vf_all!(sev, tmpv, prim, vcat(θt,σv), Val{true}, (1,1)...)

sev.EV .= 0.
sev.dEV .= 0.
sev.dEVσ .= 0.
@show parallel_solve_vf_all!(sev, vcat(θt,σv), Val{true})


rmprocs()

# using Plots
# gr()
#
# nSex = ShaleDrillingModel._nSexp(prim)
# nS = ShaleDrillingModel._nS(prim)
#
# plot(prim.wp.τmax:-1:0, sev.EV[15, 3, 1:nSex, :, 1], xaxis=(:flip,), label=string.(round.(royalty_rates,3)), xlabel="time remaining", ylabel="Emax(V)")
# plot(exp.(pspace), sev.EV[:, 3, nSex+1:nS, 4, 1], xlabel="Price", ylabel="Emax(V)")
