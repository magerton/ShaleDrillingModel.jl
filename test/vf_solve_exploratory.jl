zero!(tmpv)

let EV1 = similar(evs.EV),
    EV2 = similar(evs.EV),
    T = Float64,
    t = tmpv,
    p = prim,
    σ = σv,
    itype = (0.2,1),
    θ1 = similar(θt),
    θ2 = similar(θt),
    fdEV = similar(evs.dEV)

    zero!(evs.EV)
    zero!(evs.dEV)
    zero!(evs.dEVσ)
    zero!(tmpv)
    zero!(EV1)
    zero!(EV2)
    zero!(fdEV )

    # run through θt
    for k = 1:length(θt)
        h = peturb(θt[k])
        θ1 .= θt
        θ2 .= θt
        θ1[k] -= h
        θ2[k] += h
        hh = θ2[k] - θ1[k]

        fillflows_grad!(tmpv, prim, θ1, σv, itype...)
        solve_vf_terminal!(evs, prim)
        solve_vf_infill!(evs, tmpv, prim, false)
        learningUpdate!(evs, tmpv, prim, σv, false)
        solve_vf_explore!(evs, tmpv, prim, false)
        fdEV[:,:,k,:] .= -evs.EV

        fillflows_grad!(tmpv, prim, θ2, σv, itype...)
        solve_vf_terminal!(evs, prim)
        solve_vf_infill!(evs, tmpv, prim, false)
        learningUpdate!(evs, tmpv, prim, σv, false)
        solve_vf_explore!(evs, tmpv, prim, false)

        fdEV[:,:,k,:] .+= evs.EV
        fdEV[:,:,k,:] ./= hh
    end

        # ----------------- analytic -----------------

    fillflows_grad!(tmpv, prim, θt, σv, itype...)
    solve_vf_terminal!(evs, prim)
    solve_vf_infill!(evs, tmpv, prim, true)
    learningUpdate!(evs, tmpv, prim, σv)
    solve_vf_explore!(evs, tmpv, prim, true)

    # infill
    @views maxv, idx = findmax(abs.(fdEV[:,:,:,_nSexp(wp)+1:end].-evs.dEV[:,:,:,_nSexp(wp)+1:end]))
    sub = ind2sub(fdEV, idx)
    @show "worst value is $maxv at $sub for infill"


    # exploration
    @views maxv, idx = findmax(abs.(fdEV[:,:,:,1:_nSexp(wp)].-evs.dEV[:,:,:,1:_nSexp(wp)]))
    sub = ind2sub(fdEV, idx)
    @show "worst value is $maxv at $sub for exploratory"

    @test 0.0 < maxv < 0.1
    @test all(isfinite.(evs.dEV))
    @test all(isfinite.(fdEV))

    @test fdEV ≈ evs.dEV
    println("dEV/dθ looks ok! :)")
end






let T = Float64,
    EV1 = zeros(T, size(evs.EV)),
    EV2 = zeros(T, size(evs.EV)),
    t = tmpv,
    p = prim,
    σ = σv,
    itype = (0.2,1),
    θ1 = similar(θt),
    θ2 = similar(θt),
    fdEVσ = zeros(T, size(evs.EV)),
    fdubv = zeros(T,size(tmpv.ubVfull))

    println("testing dEV/dσ")

    zero!(evs)

    # now do σ
    h = peturb(σv)
    σ1 = σv - h
    σ2 = σv + h
    hh = σ2 - σ1

    zero!(tmpv)
    fillflows_grad!(tmpv, prim, θt, σ1, itype...)
    solve_vf_terminal!(evs, prim)
    solve_vf_infill!(evs, tmpv, prim, false)
    learningUpdate!(evs, tmpv, prim, σ1)
    fdubv .= -tmpv.ubVfull
    solve_vf_explore!(evs, tmpv, prim, false)
    fdEVσ .= -evs.EV

    zero!(tmpv)
    fillflows_grad!(tmpv, prim, θt, σ2, itype...)
    solve_vf_terminal!(evs, prim)
    solve_vf_infill!(evs, tmpv, prim, false)
    learningUpdate!(evs, tmpv, prim, σ2)
    fdubv .+= tmpv.ubVfull
    fdubv ./= hh
    solve_vf_explore!(evs, tmpv, prim, false)
    fdEVσ .+= evs.EV
    fdEVσ ./= hh

    # ----------------- analytic -----------------

    zero!(tmpv)
    fillflows_grad!(tmpv, prim, θt, σv, itype...)
    solve_vf_terminal!(evs, prim)
    solve_vf_infill!(evs, tmpv, prim, true)
    learningUpdate!(evs, tmpv, prim, σv, Val{true})
    @test fdubv ≈ tmpv.dubV_σ
    solve_vf_explore!(evs, tmpv, prim, true)

    # now test dσ
    nsexp1 = _nSexp(wp)
    fdEVσvw = @view(fdEVσ[:,:,1:nsexp1])
    @test size(fdEVσvw) == size(evs.dEVσ)
    @views maxv, idx = findmax(abs.(fdEVσvw.-evs.dEVσ))
    sub = ind2sub(fdEVσvw, idx)
    @show "worst value is $maxv at $sub for dσ"
    @show extrema(fdEVσvw)
    @show extrema(evs.dEVσ)

    maximum(abs.((evs.dEVσ .- fdEVσvw)[:,:,2:end-1]))

    @test all(isfinite.(evs.dEVσ))
    @test all(isfinite.(fdEVσvw))
    @test 0.0 < maxv < 0.1

    @test fdEVσvw ≈ evs.dEVσ
    println("dEV/dσ looks ok! :)")
end
