println("testing dEV_σ")

# ------------------------------- exploratory ----------------------------------

tmpv.ubVfull .= 0.
tmpv.dubVfull .= 0.
tmpv.dubV_σ .= 0.

let fduσbv = zeros(size(tmpv.ubVfull)),
    ubv1   = zeros(size(tmpv.ubVfull)),
    ubv2   = zeros(size(tmpv.ubVfull)),
    uin1   = zeros(size(tmpv.uin)),
    uin1   = zeros(size(tmpv.uin)),
    uin2   = zeros(size(tmpv.uin)),
    uex1   = zeros(size(tmpv.uex)),
    uex2   = zeros(size(tmpv.uex)),
    EV1 = similar(evs.EV),
    EV2 = similar(evs.EV),
    vpos = 1,
    T = Float64,
    h = max( abs(σv), one(T) ) * cbrt(eps(T)),
    t = tmpv,
    p = prim,
    σ = σv,
    fdEVσ = similar(evs.dEV_σ),
    roy = 0.2

    for (vpos,v) in enumerate(vspace)

        fillflows!(tmpv, p, θt, σ, roy, v, -h)
        solve_vf_infill!(evs, tmpv, prim)
        learningUpdate!(tmpv, evs, prim, σv, v, -h)
        uin1 .= tmpv.uin
        uex1 .= tmpv.uex
        solve_vf_explore!(evs.EV, uex1, tmpv.ubVfull, tmpv.lse, tmpv.tmp, p.wp, p.Πz, tmpv.βΠψ, β)
        EV1 .= evs.EV
        ubv1 .= tmpv.ubVfull
        @views ubv1[:,:,1] .= β .* EV1[:,:,1]

        fillflows!(tmpv, p, θt, σ, roy, v, +h)
        solve_vf_infill!(evs, tmpv, prim)
        learningUpdate!(tmpv, evs, prim, σv, v, +h)
        uin2 .= tmpv.uin
        uex2 .= tmpv.uex
        @test all(uin1 .== uin2)
        @test !all(uex1 .== uex2)
        solve_vf_explore!(evs.EV, uex2, tmpv.ubVfull, tmpv.lse, tmpv.tmp, p.wp, p.Πz, tmpv.βΠψ, β)
        EV2 .= evs.EV
        ubv2 .= tmpv.ubVfull
        @views ubv2[:,:,1] .= β .* EV2[:,:,1]

        fillflows_grad!(tmpv, p, θt, σ, roy)
        @test (uex2 .- uex1) ./ (2.0 .* h) ≈ tmpv.duexσ[:,:,vpos,:]
        solve_vf_infill!(evs, tmpv, prim)
        learningUpdate!(tmpv, evs, prim, σ, true)
        solve_vf_explore!(evs, tmpv, prim)
        tmpv.dubV_σ[:,:,:,1] .= β .* evs.dEV_σ[:,:,:,1]

        fduσbv .= (ubv2 .- ubv1) ./ (2.0.*h)
        @test fduσbv ≈ tmpv.dubV_σ[:,:,vpos,:]

        nsexp1 = ShaleDrillingModel._nSexp(prim)+1
        @views fdEVσ[:,:,vpos,:] .= (EV2[:,:,1:nsexp1] .- EV1[:,:,1:nsexp1]) ./ (2.0.*h)
    end

    @test fdEVσ ≈ evs.dEV_σ
end

println("test of dEV_σ passed!")

























# end

# -----------------------------------------------------------------
