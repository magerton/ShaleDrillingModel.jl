
@testset "Perfect Info" begin
    T = Float64
    roy = 0.2
    geoid = 2
    itype = (geoid, roy,)
    nsexp = ShaleDrillingModel._nSexp(wp)
    dograd = false

    thet0 = [-3.0, -5.45746, -0.9, ] # α₀, α_cost, α_extend
    sig0 = 60.0
    f0 = StaticDrillingPayoff(DrillingRevenue(Constrained(log_ogip=0.64, α_ψ=0.35, α_t=0.025), NoTrend(), NoTaxes(), Learn(), WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())
    prim0 = dcdp_primitives(f0, β, wp, zspace, Πp, ψspace)

    thet1 = thet0
    sig1 = 0.7
    f1 = StaticDrillingPayoff(DrillingRevenue(Constrained(log_ogip=0.64, α_ψ=0.35, α_t=0.025), NoTrend(), NoTaxes(), PerfectInfo(), WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())
    prim1 = dcdp_primitives(f1, β, wp, zspace, Πp, ψspace)

    evs = dcdp_Emax(prim0)
    tmpv = dcdp_tmpvars(prim0)
    EV0 = similar(evs.EV)

    # solve 0
    solve_vf_all!(evs, tmpv, prim0, thet0, sig0, itype, dograd)
    EV0 .= evs.EV

    # solve 1
    solve_vf_all!(evs, tmpv, prim1, thet1, sig1, itype, dograd)

    @test evs.EV ≈ EV0
end


@testset "Max Learning" begin
    T = Float64
    roy = 0.2
    geoid = 2
    itype = (geoid, roy,)
    nsexp = ShaleDrillingModel._nSexp(wp)
    dograd = false

    thet0 = [-3.0, -5.45746, -0.9, ] # α₀, α_cost, α_extend
    sig0 = -60.0
    f0 = StaticDrillingPayoff(DrillingRevenue(Constrained(log_ogip=0.64, α_ψ=0.35, α_t=0.025), NoTrend(), NoTaxes(), Learn(), WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())
    prim0 = dcdp_primitives(f0, β, wp, zspace, Πp, ψspace)

    thet1 = thet0
    sig1 = 0.7
    f1 = StaticDrillingPayoff(DrillingRevenue(Constrained(log_ogip=0.64, α_ψ=0.35, α_t=0.025), NoTrend(), NoTaxes(), MaxLearning(), WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())
    prim1 = dcdp_primitives(f1, β, wp, zspace, Πp, ψspace)

    evs = dcdp_Emax(prim0)
    tmpv = dcdp_tmpvars(prim0)
    EV0 = similar(evs.EV)

    # solve 0
    solve_vf_all!(evs, tmpv, prim0, thet0, sig0, itype, dograd)
    EV0 .= evs.EV

    # solve 1
    solve_vf_all!(evs, tmpv, prim1, thet1, sig1, itype, dograd)

    @test evs.EV ≈ EV0
end



@testset "NoLearn" begin
    T = Float64
    roy = 0.2
    geoid = 2
    itype = (geoid, roy,)
    nsexp = ShaleDrillingModel._nSexp(wp)
    dograd = false

    α0 = -3.0
    αψ0 = 0.35
    sig0 = 0.7
    thet0 = [α0, -5.45746, -0.9, ] # α₀, α_cost, α_extend
    ρ0 = _ρ(sig0)
    f0 = StaticDrillingPayoff(DrillingRevenue(Constrained(log_ogip=0.64, α_ψ=αψ0, α_t=0.025), NoTrend(), NoTaxes(), NoLearn(), WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())
    prim0 = dcdp_primitives(f0, β, wp, zspace, Πp, ψspace)

    α1 = α0 + 0.5*αψ0^2*(1-ρ0^2)
    αψ1 = αψ0*ρ0
    thet1 = vcat(α1, thet0[2:end])
    sig1 = 60.0
    f1 = StaticDrillingPayoff(DrillingRevenue(Constrained(log_ogip=0.64, α_ψ=αψ1, α_t=0.025), NoTrend(), NoTaxes(), PerfectInfo(), WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())
    prim1 = dcdp_primitives(f1, β, wp, zspace, Πp, ψspace)

    evs = dcdp_Emax(prim0)
    tmpv = dcdp_tmpvars(prim0)
    EV0 = similar(evs.EV)

    # solve 0
    solve_vf_all!(evs, tmpv, prim0, thet0, sig0, itype, dograd)
    EV0 .= evs.EV

    # solve 1
    solve_vf_all!(evs, tmpv, prim1, thet1, sig1, itype, dograd)

    @test @views evs.EV[:,:, end_lrn(wp)+1 : end        ] ≈ EV0[:,:, end_lrn(wp)+1 : end]
    @test @views evs.EV[:,:, end_ex0(wp)+1 : end_lrn(wp)] ≈ EV0[:,:, end_ex0(wp)+1 : end_lrn(wp)]
    @test @views evs.EV[:,:,             1 : end_ex0(wp)] ≈ EV0[:,:,             1 : end_ex0(wp)]
end




@testset "NoRoyalty" begin
    T = Float64
    roy = 0.2
    geoid = 2
    itype = (geoid, roy,)
    nsexp = ShaleDrillingModel._nSexp(wp)
    dograd = false

    sig0 = 0.7
    thet0 = [-3.0, -5.45746, -0.9, ] # α₀, α_cost, α_extend
    f0 = StaticDrillingPayoff(DrillingRevenue(Constrained(log_ogip=0.64, α_ψ=0.35, α_t=0.025), NoTrend(), NoTaxes(), Learn(), WithRoyalty()), DrillingCost_constant(), ExtensionCost_Constant())
    f1 = StaticDrillingPayoff(DrillingRevenue(Constrained(log_ogip=0.64, α_ψ=0.35, α_t=0.025), NoTrend(), NoTaxes(), Learn(), NoRoyalty()),   DrillingCost_constant(), ExtensionCost_Constant())
    prim0 = dcdp_primitives(f0, β, wp, zspace, Πp, ψspace)
    prim1 = dcdp_primitives(f1, β, wp, zspace, Πp, ψspace)

    evs = dcdp_Emax(prim0)
    tmpv = dcdp_tmpvars(prim0)
    EV0 = similar(evs.EV)

    # solve 0
    solve_vf_all!(evs, tmpv, prim0, thet0, sig0, (geoid, 0.0,), dograd)
    EV0 .= evs.EV

    # solve 1
    solve_vf_all!(evs, tmpv, prim1, thet0, sig0, (geoid, 0.2), dograd)

    @test evs.EV ≈ EV0
end
