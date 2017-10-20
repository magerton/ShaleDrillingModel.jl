
let EV = evs.EV, dEV = evs.dEV, dEV_σ = evs.dEV_σ,
    t = tmpv,p = prim,
    idxs = [ShaleDrillingModel.explore_state_inds(wp)..., ShaleDrillingModel.infill_state_inds(wp)..., ShaleDrillingModel.terminal_state_ind(wp)...]

    @test idxs ⊆ 1:length(wp)
    @test 1:length(wp) ⊆ idxs

    # full VFI
    ShaleDrillingModel.solve_vf_terminal!(EV, dEV, dEV_σ)
    ShaleDrillingModel.solve_vf_infill!(EV, t.uin, t.ubVfull, t.lse, t.tmp, t.IminusTEVp, p.wp, p.Πz, p.β)
    ShaleDrillingModel.solve_vf_infill!(EV, dEV, t.uin, t.duin, t.ubVfull, t.dubVfull, t.lse, t.tmp, t.IminusTEVp, p.wp, p.Πz, p.β)
    @test !all(EV .== 0.)
end

# ---------------- Regime 1 VFI and PFI ------------------

let roy = 0.2

    evs.EV .= 0.0
    fillflows_grad!(tmpv, prim, θt, σv, roy)
    solve_vf_terminal!(evs)
    solve_vf_infill!(evs, tmpv, prim, false)

    # using Plots
    # gr()
    #
    # plot(exp.(pspace), evs.EV[:, 3, end-1:-1:end-6])
end
