# zero!(tmpv)

@testset "Exploratory EV gradient" begin
    T = Float64
    t = tmpv
    p = prim
    σ = σv
    roy = 0.2
    geoid = 2
    itype = (geoid, roy,)
    θ1 = similar(θt)
    θ2 = similar(θt)
    EV1 = similar(evs.EV)
    EV2 = similar(evs.EV)
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

        solve_vf_terminal!(evs, prim)
        solve_vf_infill!( evs, tmpv, prim, θ1, σv, false, itype)
        learningUpdate!(  evs, tmpv, prim,     σv, false)
        solve_vf_explore!(evs, tmpv, prim, θ1, σv, false, itype)
        fdEV[:,:,k,:] .= -evs.EV

        solve_vf_terminal!(evs, prim)
        solve_vf_infill!( evs, tmpv, prim, θ2, σv, false, itype)
        learningUpdate!(  evs, tmpv, prim,     σv, false)
        solve_vf_explore!(evs, tmpv, prim, θ2, σv, false, itype)

        fdEV[:,:,k,:] .+= evs.EV
        fdEV[:,:,k,:] ./= hh
    end

    # ----------------- analytic -----------------

    solve_vf_terminal!(evs, prim)
    solve_vf_infill!( evs, tmpv, prim, θt, σv, true, itype)
    learningUpdate!(  evs, tmpv, prim,     σv, true)
    solve_vf_explore!(evs, tmpv, prim, θt, σv, true, itype)

    # check infill + learning portion gradient
    @views maxv, idx = findmax(abs.(fdEV[:,:,:,_nSexp(wp)+1:end].-evs.dEV[:,:,:,_nSexp(wp)+1:end]))
    println("worst value is $maxv at $(CartesianIndices(fdEV)[idx]) for infill")

    # check exploration portion of gradient
    @views maxv, idx = findmax(abs.(fdEV[:,:,:,1:_nSexp(wp)].-evs.dEV[:,:,:,1:_nSexp(wp)]))
    println("worst value is $maxv at $(CartesianIndices(fdEV)[idx]) for exploratory")

    @test 0.0 < maxv < 0.1
    @test all(isfinite.(evs.dEV))
    @test all(isfinite.(fdEV))

    @test fdEV ≈ evs.dEV || maxv < 1e-5
    println("dEV/dθ looks ok! :)")
end





@testset "dEV/dσ" begin
    let T = Float64,
        EV1 = zeros(T, size(evs.EV)),
        EV2 = zeros(T, size(evs.EV)),
        t = tmpv,
        p = prim,
        nsexp1 = _nSexp(wp),
        σ = σv,
        roy = 0.2,
        geoid = 2,
        itype = (geoid, roy,),
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
        solve_vf_terminal!(evs, prim)
        solve_vf_infill!( evs, tmpv, prim, θt, σ1, false, itype)
        learningUpdate!(  evs, tmpv, prim,     σ1, false)
        # fdubv .= -tmpv.ubVfull
        solve_vf_explore!(evs, tmpv, prim, θt, σ1, false, itype)
        fdEVσ .= -evs.EV

        zero!(tmpv)
        solve_vf_terminal!(evs, prim)
        solve_vf_infill!( evs, tmpv, prim, θt, σ2, false, itype)
        learningUpdate!(  evs, tmpv, prim,     σ2, false)
        # fdubv .+= tmpv.ubVfull
        # fdubv ./= hh
        solve_vf_explore!(evs, tmpv, prim, θt, σ2, false, itype)
        fdEVσ .+= evs.EV
        fdEVσ ./= hh

        # ----------------- analytic -----------------

        zero!(tmpv)
        solve_vf_infill!( evs, tmpv, prim, θt, σv, true, itype)
        learningUpdate!(  evs, tmpv, prim,     σv, true)

        # # test dubV_σ
        # maxv, idx = findmax(abs.(fdubv.-tmpv.dubV_σ))
        # @test fdubv ≈ tmpv.dubV_σ || maxv < 6.0e-7
        # println("worst value is $maxv at $(CartesianIndices(fdubv)[idx]) for dubV_σ")
        # @show extrema(fdubv)
        # @show extrema(tmpv.dubV_σ)

        # now test dσ
        solve_vf_explore!(evs, tmpv, prim, θt, σv, true, itype)
        fdEVσvw = @view(fdEVσ[:,:,1:nsexp1])
        @test size(fdEVσvw) == size(evs.dEVσ)
        maxv, idx = findmax(abs.(fdEVσvw.-evs.dEVσ))
        println("worst value is $maxv at $(CartesianIndices(fdEVσvw)[idx]) for dσ")
        @show extrema(fdEVσvw)
        @show extrema(evs.dEVσ)

        @show maximum(abs.((evs.dEVσ .- fdEVσvw)[:,:,2:end-1]))

        @test all(isfinite.(evs.dEVσ))
        @test all(isfinite.(fdEVσvw))
        @test 0.0 < maxv < 0.1

        @test fdEVσvw ≈ evs.dEVσ # || maxv < 1.5e-6
        println("dEV/dσ looks ok! :)")
    end
end
