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
    h = cbrt(eps(Float64)),
    t = tmpv,
    p = prim,
    σ = σv,
    fdEV = similar(evs.EV),
    roy = 0.2

    v = vspace[vpos]

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
    learningUpdate!(tmpv, evs, prim, σv, v, h)
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

    fduσbv .= (ubv2 .- ubv1) ./ (2.0.*h)
    @show findmax(abs.(fduσbv .-  tmpv.dubV_σ[:,:,vpos,:]))
    # only test 2:end b/c we don't update ubV at the end...
    @test tmpv.dubV_σ[:,:,vpos, 2:end] ≈ fduσbv[:,:,2:end]

    fdEV .= (EV2 .- EV1) ./ (2.0.*h)

    fdEVvw = @view(fdEV[:,:,1:ShaleDrillingModel._nSexp(p)]) # explore_state_inds(wp)[end:-1:1]])
    dEVσvw = @view(evs.dEV_σ[:,:,vpos,1:end-1])
    @test fdEVvw ≈ dEVσvw
end

# -----------------------------------------------------------------
