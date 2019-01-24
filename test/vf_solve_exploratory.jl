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
    nsexp = ShaleDrillingModel._nSexp(wp)

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
    @views maxv, idx = findmax(abs.(fdEV[:,:,:,nsexp+1:end].-evs.dEV[:,:,:,nsexp+1:end]))
    println("worst value is $maxv at $(CartesianIndices(fdEV)[idx]) for infill")

    # check exploration portion of gradient
    @views maxv, idx = findmax(abs.(fdEV[:,:,:,1:nsexp].-evs.dEV[:,:,:,1:nsexp]))
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
        nsexp1 = ShaleDrillingModel._nSexp(wp),
        σ = σv,
        roy = 0.2,
        geoid = 2,
        itype = (geoid, roy,),
        θ1 = similar(θt),
        θ2 = similar(θt),
        fdEVσ = zeros(T, size(evs.EV)),
        fdubv = zeros(T,size(tmpv.ubVfull)),
        fdEVσvw = @view(fdEVσ[:,:,1:nsexp1])

        println("testing dEV/dσ")

        # now do σ
        h = peturb(σv)
        σ1 = σv - h
        σ2 = σv + h
        hh = σ2 - σ1

        dograd = false

        zero!(evs)
        zero!(tmpv)
        solve_vf_all!(evs, tmpv, prim, θt, σ1, itype, dograd)
        # solve_vf_terminal!(evs, prim)
        # solve_vf_infill!( evs, tmpv, prim, θt, σ1, dograd, itype)
        # learningUpdate!(  evs, tmpv, prim,     σ1, dograd)
        # solve_vf_explore!(evs, tmpv, prim, θt, σ1, dograd, itype)

        fdEVσ .= -evs.EV

        zero!(evs)
        zero!(tmpv)
        solve_vf_all!(evs, tmpv, prim, θt, σ2, itype, dograd)
        # solve_vf_terminal!(evs, prim)
        # solve_vf_infill!( evs, tmpv, prim, θt, σ2, dograd, itype)
        # learningUpdate!(  evs, tmpv, prim,     σ2, dograd)
        # solve_vf_explore!(evs, tmpv, prim, θt, σ2, dograd, itype)

        fdEVσ .+= evs.EV
        fdEVσ ./= hh

        # ----------------- analytic -----------------

        dograd = true

        zero!(evs)
        zero!(tmpv)
        solve_vf_all!(evs, tmpv, prim, θt, σv, itype, dograd)
        # solve_vf_terminal!(evs, prim)
        # solve_vf_infill!(  evs, tmpv, prim, θt, σv, dograd, itype)
        # learningUpdate!(   evs, tmpv, prim,     σv, dograd)
        # solve_vf_explore!( evs, tmpv, prim, θt, σv, dograd, itype)

        @test size(fdEVσvw) == size(evs.dEVσ)
        @test all(isfinite.(evs.dEVσ))
        @test all(isfinite.(fdEVσvw))
        @show extrema(fdEVσvw)
        @show extrema(evs.dEVσ)

        println("Testing evs.dEVσ[:,:,2:end] .- fdEVσvw[:,:,2:end]")
        @show maximum(abs.((evs.dEVσ .- fdEVσvw)[:,:,2:end]))
        @test @views fdEVσvw[:,:,2:end] ≈ evs.dEVσ[:,:,2:end]

        maxv, idx = findmax(abs.(fdEVσvw.-evs.dEVσ))
        println("worst value is $maxv at $(CartesianIndices(fdEVσvw)[idx]) for dσ")
        @test 0.0 < maxv < 0.1
        @test @views fdEVσvw[:,:,1] ≈ evs.dEVσ[:,:,1]

        @test fdEVσvw ≈ evs.dEVσ # || maxv < 1.5e-6
        println("dEV/dσ looks ok! :)")
    end
end
