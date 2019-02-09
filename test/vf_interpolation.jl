@testset "VF Inteprolation" begin
    let prim = dcdp_primitives(flowfuncname, β, wp, zspace, Πp, ψspace),
        tmpv = dcdp_tmpvars(prim),
        royalty_rates = [0.25,],
        geology_types = 2:2

        sev = SharedEV([1,], prim, geology_types, royalty_rates)
        EVcopy = similar(sev.EV)
        EVcopy2 = similar(sev.EV)

        serial_solve_vf_all!(sev, tmpv, prim, θfull, false)
        EVcopy .= sev.EV
        EVcopy2 .= sev.EV
        @test !(EVcopy === sev.EV)

        EVcopy .= sev.EV
        @test all(EVcopy .== sev.EV)
        spec = (BSpline(Quadratic(InPlace())), BSpline(Quadratic(InPlace())), BSpline(Quadratic(InPlace())), NoInterp(), NoInterp(), NoInterp(),)
        sitp_test = interpolate!(EVcopy, spec, OnCell() )
        @test sitp_test.coefs === EVcopy
        @test !all(sitp_test.coefs .== sev.EV)

        sitp = ItpSharedEV(sev, prim)
        sitp.itypes

        serial_prefilterByView!(sev, sitp)

        @test all(sitp_test.coefs .== sev.EV)
        @test !all(sev.EV .== EVcopy2)
    end
end
