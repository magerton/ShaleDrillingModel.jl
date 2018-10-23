
let prim = dcdp_primitives(:addlin, β, wp, zspace, Πp1, ψspace),
    tmpv = dcdp_tmpvars(prim),
    sev = SharedEV([1,], prim, [1./8.], 1:1),
    EVcopy = similar(sev.EV),
    EVcopy2 = similar(sev.EV)

    serial_solve_vf_all!(sev, tmpv, prim, θfull, Val{false})
    EVcopy .= sev.EV
    EVcopy2 .= sev.EV
    @test !(EVcopy === sev.EV)

    EVcopy .= sev.EV
    @test all(EVcopy .== sev.EV)
    sitp_test = interpolate!(EVcopy, (BSpline(Quadratic(InPlace())), BSpline(Quadratic(InPlace())), NoInterp(), NoInterp(), NoInterp()), OnCell() )
    @test sitp_test.coefs === EVcopy
    @test !all(sitp_test.coefs .== sev.EV)

    sitp = ItpSharedEV(sev, prim)
    sitp.itypes

    serial_prefilterByView!(sev, sitp)

    @test all(sitp_test.coefs .== sev.EV)
    @test !all(sev.EV .== EVcopy2)
end
