
let EV = evs.EV,
    dEV = evs.dEV,
    dEVσ = evs.dEVσ,
    t = tmpv,
    p = prim

    # full VFI
    ShaleDrillingModel.solve_vf_terminal!(EV, dEV, dEVσ, wp)
    ShaleDrillingModel.solve_vf_infill!(EV, t.uin, t.ubVfull, t.lse, t.tmp, t.IminusTEVp, p.wp, p.Πz, p.β)
    ShaleDrillingModel.solve_vf_infill!(EV, dEV, t.uin, t.duin, t.ubVfull, t.dubVfull, t.lse, t.tmp, t.IminusTEVp, p.wp, p.Πz, p.β)
    @test !all(EV .== 0.)
end

# ---------------- Regime 1 VFI and PFI - test gradient ------------------

let θ1 = similar(θt), θ2 = similar(θt), fdEV = similar(evs.dEV), itype = (0.2,1), tmp=tmpv
    for k = 1:length(θt)
        h = peturb(θt[k])
        θ1 .= θt
        θ2 .= θt
        θ1[k] -= h
        θ2[k] += h
        hh = θ2[k] - θ1[k]

        fillflows_grad!(tmpv, prim, θ1, σv, itype...)
        solve_vf_terminal!(evs, prim)
        solve_vf_infill!(evs, tmp, prim, false)
        fdEV[:,:,k,:] .= -evs.EV

        fillflows_grad!(tmpv, prim, θ2, σv, itype...)
        solve_vf_terminal!(evs, prim)
        solve_vf_infill!(evs, tmp, prim, false)
        fdEV[:,:,k,:] .+= evs.EV
        fdEV[:,:,k,:] ./= hh
    end
    fillflows_grad!(tmpv, prim, θt, σv, itype...)
    solve_vf_terminal!(evs, prim)
    solve_vf_infill!(evs, tmp, prim, true)

    @views maxv, idx = findmax(abs.(fdEV[:,:,2:end,:].-evs.dEV[:,:,2:end,:]))
    @views sub = ind2sub(fdEV[:,:,2:end,:], idx)
    @show "worst value is $maxv at $sub for dθ[2:end]"

    maxv, idx = findmax(abs.(fdEV.-evs.dEV))
    sub = ind2sub(fdEV, idx)
    @show "worst value is $maxv at $sub"
    # @test fdEV ≈ evs.dEV
    @test 0.0 < maxv < 1.0
    @test all(isfinite.(evs.dEV))
    @test all(isfinite.(fdEV))
end
