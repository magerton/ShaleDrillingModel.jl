# ---------------- Regime 2 VFI and PFI ------------------

# try a single VFI
let EVtmp = zeros(nz,nψ)
    ShaleDrillingModel.vfit!(EVtmp, ubV, lse, tmp, Πz)
    @show extrema(EVtmp .- EV0) .* β ./ (1.0-β)
end

# try VFI for inf horizon
let EVvfit = zeros(eltype(EV0), size(EV0)),
    EVpfit = zeros(eltype(EV0), size(EV0)),
    dEVvfit = zeros(eltype(dEV0), size(dEV0)),
    dEVpfit = zeros(eltype(dEV0), size(dEV0))

    # solve with VFI only
    EV .= 0.0
    ubV .= @view(uin0[:,:,1:dmaxp1])
    @show ShaleDrillingModel.solve_inf_vfit!(EVvfit, ubV, lse, tmp, Πz, β, maxit=5000, vftol=1e-12)

    # solve with hybrid iteration (12 VFit steps + more PFit)
    ubV .= @view(uin0[:,:,1:dmaxp1])
    ShaleDrillingModel.pfit!(EV0, ubV, lse, tmp, IminusTEVp, Πz, β)
    @show ShaleDrillingModel.solve_inf_vfit!(EVpfit, ubV, lse, tmp,             Πz, β, maxit=12, vftol=1.0)
    @show ShaleDrillingModel.solve_inf_pfit!(EVpfit, ubV, lse, tmp, IminusTEVp, Πz, β; maxit=20, vftol=1e-11)
    @test EVpfit ≈ EVvfit

    # update ubV and make inf horizon derivatives
    ubV .= @view(uin0[:,:,1:dmaxp1])
    ubV[:,:,1] .+=  β .* EVpfit
    dubV .= @view(duin0[:,:,:,1:dmaxp1])
    ShaleDrillingModel.gradinf!(dEVpfit, ubV, dubV, lse, tmp, IminusTEVp, Πz, β)  # note: destroys ubV

    # update dubV with gradinf! results & test that when we run VFI grad, we get the same thing back.
    # Note: since ubV destroyed, re-make
    ubV .= @view(uin0[:,:,1:dmaxp1])
    ubV[:,:,1] .+= β .* EVpfit
    dubV .= @view(duin0[:,:,:,1:dmaxp1])
    dubV[:,:,:,1] .+= β .* dEVpfit
    ShaleDrillingModel.vfit!(EVvfit, dEVvfit, ubV, dubV, lse, tmp, Πz)
    @test EVpfit ≈ EVvfit
    @test dEVpfit ≈ dEVvfit
end
