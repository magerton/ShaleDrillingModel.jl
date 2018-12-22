@testset "Make sure that EV is filled" begin
    EV = evs.EV
    dEV = evs.dEV
    dEVσ = evs.dEVσ
    t = tmpv
    p = prim

    @test !all(t.uin .== 0.0)
    @test !all(t.duin .== 0.0)

    # test that full VFI does something
    fill!(EV, 0.0)
    ShaleDrillingModel.solve_vf_terminal!(EV, dEV, dEVσ, wp)
    solve_vf_infill!(evs, t, p, false)
    @test !all(EV .== 0.0)

    # test that full VFI does something and gets gradient
    fill!(EV, 0.0)
    fill!(dEV, 0.0)
    solve_vf_infill!(evs, t, p, true)
    @test !all(EV .== 0.0)
    @test !all(dEV .== 0.0)
end

# ---------------- Regime 1 VFI and PFI - test gradient ------------------

@testset "Gradient of EV for infill" begin
    θ1 = similar(θt)
    θ2 = similar(θt)
    fdEV = similar(evs.dEV)
    roy = 0.25
    geoid = 2
    itype = (geoid, roy,)
    tmp = tmpv

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
    @views sub = CartesianIndices(fdEV[:,:,2:end,:])[idx]
    println("worst value is $maxv at $sub for dθ[2:end]")

    maxv, idx = findmax(abs.(fdEV.-evs.dEV))
    sub = CartesianIndices(fdEV)[idx]
    println("worst value is $maxv at $sub")
    # @test fdEV ≈ evs.dEV
    @test 0.0 < maxv < 1.0
    @test all(isfinite.(evs.dEV))
    @test all(isfinite.(fdEV))
end
