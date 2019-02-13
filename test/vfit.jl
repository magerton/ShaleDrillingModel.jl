let EV         = evs.EV,
    dEV        = evs.dEV,
    ubVfull    = tmpv.ubVfull,
    dubVfull   = tmpv.dubVfull,
    lse        = tmpv.lse,
    tmp        = tmpv.tmp,
    IminusTEVp = tmpv.IminusTEVp,
    p          = prim,
    σ          = σv,
    wp         = prim.wp,
    Πz         = prim.Πz,
    β          = prim.β,
    fdEV       = similar(evs.dEV),
    θ1         = similar(θt),
    θ2         = similar(θt),
    roy        = 0.25,
    geoid      = 4.7,
    itype      = (geoid, roy,),
    nS         = length(prim.wp),
    i          = nS-1,
    idxd       = ShaleDrillingModel.dp1space(wp,i),
    idxs       = collect(ShaleDrillingModel.sprimes(wp,i)),
    horzn      = ShaleDrillingModel._horizon(wp,i)

    @views ubV = ubVfull[:,:,idxd]
    @views dubV = dubVfull[:,:,:,idxd]
    @views EV0 = EV[:,:,i]
    @views EV1 = EV[:,:,idxs]
    @views dEV0 = dEV[:,:,:,i]
    @views dEV1 = dEV[:,:,:,idxs]
    @views fdEV0 = fdEV[:,:,:,i]


    @testset "Check finite horizon gradient" begin
        for k = 1:length(θt)
            h = peturb(θt[k])
            θ1 .= θt
            θ2 .= θt
            θ1[k] -= h
            θ2[k] += h
            hh = θ2[k] - θ1[k]

            fill!(EV, 0.0)
            fillflows!(ubV, flow, p, θ1, σ, i, itype...)
            ubV .+= β .* EV1
            ShaleDrillingModel.vfit!(EV0, ubV, lse, tmp, Πz)
            fdEV0[:,:,k] .= -EV0

            fill!(EV, 0.0)
            fillflows!(ubV, flow, p, θ2, σ, i, itype...)
            ubV .+= β .* EV1
            ShaleDrillingModel.vfit!(EV0, ubV, lse, tmp, Πz)
            fdEV0[:,:,k] .+= EV0
            fdEV0[:,:,k] ./= hh
        end

        fill!(EV, 0.0)
        fill!(dEV, 0.0)
        fillflows!(ubV, flow, p, θt, σ, i, itype...)
        fillflows_grad!(dubV, flowdθ, p, θt, σ, i, itype...)
        ubV .+= β .* EV1
        dubV .+= β .* dEV1
        ShaleDrillingModel.vfit!(EV0, dEV0, ubV, dubV, lse, tmp, Πz)

        println("extrema(dEV0) = $(extrema(dEV0))")
        @views maxv, idx = findmax(abs.(fdEV0 .- dEV0))
        @views sub = CartesianIndices(fdEV0)[idx]
        println("worst value is $maxv at $sub for dθ")

        @test 0.0 < maxv < 1.0
        @test all(isfinite.(dEV0))
        @test all(isfinite.(fdEV0))
        @test fdEV0 ≈ dEV0
    end

    @testset "Check vfit/pfit for infinite horizon" begin
        vfEV0 = zeros(size(EV0))

        fill!(EV, 0.0)
        fillflows!(ubV, flow, p, θt, σ, i, itype...)
        ubV .+= β .* EV1
        converged, iter, bnds = ShaleDrillingModel.solve_inf_vfit!(EV0, ubV, lse, tmp, Πz, β; maxit=1000, vftol=1e-10)
        println("vfit done. converged = $converged after $iter iterations. error bounds are $bnds")
        vfEV0 .= EV0

        fill!(EV, 0.0)
        fillflows!(ubV, flow, p, θt, σ, i, itype...)
        ubV .+= β .* EV1
        converged, iter, bnds = ShaleDrillingModel.solve_inf_vfit_pfit!(EV0, ubV, lse, tmp, IminusTEVp, Πz, β; vftol=1e-10, maxit0=20, maxit1=40)
        println("pfit done. converged = $converged after $iter iterations. error bounds are $bnds")

        @views maxv, idx = findmax(abs.(vfEV0 .- EV0))
        @views sub = CartesianIndices(EV0)[idx]
        println("worst value is $maxv at $sub for vfit vs pfit")
        @test EV0 ≈ vfEV0
    end

    @testset "Check gradient for infinite horizon" begin


        for k = 1:length(θt)
            h = peturb(θt[k])
            θ1 .= θt
            θ2 .= θt
            θ1[k] -= h
            θ2[k] += h
            hh = θ2[k] - θ1[k]

            fill!(EV, 0.0)
            fillflows!(ubV, flow, p, θ1, σ, i, itype...)
            ubV .+= β .* EV1
            ShaleDrillingModel.solve_inf_vfit_pfit!(EV0, ubV, lse, tmp, IminusTEVp, Πz, β; vftol=1e-10, maxit0=20, maxit1=40)
            fdEV0[:,:,k] .= -EV0

            fill!(EV, 0.0)
            fillflows!(ubV, flow, p, θ2, σ, i, itype...)
            ubV .+= β .* EV1
            ShaleDrillingModel.solve_inf_vfit_pfit!(EV0, ubV, lse, tmp, IminusTEVp, Πz, β; vftol=1e-10, maxit0=20, maxit1=40)
            fdEV0[:,:,k] .+= EV0
            fdEV0[:,:,k] ./= hh
        end

        fill!(EV, 0.0)
        fill!(dEV, 0.0)
        fillflows!(ubV, flow, p, θt, σ, i, itype...)
        fillflows_grad!(dubV, flowdθ, p, θt, σ, i, itype...)
        ubV .+= β .* EV1
        dubV .+= β .* dEV1
        ShaleDrillingModel.solve_inf_vfit_pfit!(EV0, ubV, lse, tmp, IminusTEVp, Πz, β; vftol=1e-10, maxit0=20, maxit1=40)
        ubV[:,:,1] .= β .* EV0
        ShaleDrillingModel.gradinf!(dEV0, ubV, dubV, lse, tmp, IminusTEVp, Πz, β)   # note: destroys ubV & dubV

        println("extrema(dEV0) = $(extrema(dEV0))")
        @views maxv, idx = findmax(abs.(fdEV0 .- dEV0))
        @views sub = CartesianIndices(fdEV0)[idx]
        println("worst value is $maxv at $sub for dθ")

        @test 0.0 < maxv < 1.0
        @test all(isfinite.(dEV0))
        @test all(isfinite.(fdEV0))
        @test fdEV0 ≈ dEV0

    end
end
