let dEV = Array{Float64}(size(shev.dEV)),
    dEVσ = Array{Float64}(size(shev.dEVσ)),
    fdEV = similar(dEV),
    fdEVσ = similar(dEVσ),
    θfull = vcat(θt,σv),
    θ1 = similar(θfull),
    θ2 = similar(θfull)

    println("testing parallel solving of EV")

    shev.EV .= 0.0
    shev.dEV .= 0.0
    shev.dEVσ .= 0.0

    # serial_solve_vf_all!(shev, tmpv, prim, θfull, Val{true})
    s = parallel_solve_vf_all!(shev, vcat(θt,σv), Val{true})
    @show fetch.(s)

    @test maximum(abs.(shev.EV)) < 1e8
    @test maximum(abs.(shev.dEV)) < 1e8
    @test maximum(abs.(shev.dEVσ)) < 1e8

    dEV .= shev.dEV

    for k in 1:length(θt)
        T = Float64
        θ1 .= θfull
        θ2 .= θfull
        h = max( abs(θfull[k]), one(T) ) * cbrt(eps(T))
        θ1[k] -= h
        θ2[k] += h
        hh = θ2[k] - θ1[k]
        parallel_solve_vf_all!(shev, θ1, Val{false})
        fdEV[:,:,k,:,:,:] .= -shev.EV
        parallel_solve_vf_all!(shev, θ2, Val{false})
        fdEV[:,:,k,:,:,:] .+= shev.EV
        fdEV[:,:,k,:,:,:] ./= hh
    end


    for CI in CartesianRange(length.(shev.itypes))
        @test dEV[:,:,:,:,CI] ≈ fdEV[:,:,:,:,CI]
    end

    println("dEV seems ok! :)")
end
