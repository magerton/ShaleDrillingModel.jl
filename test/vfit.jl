let EV         = evs.EV,
    dEV        = evs.dEV,
    ubVfull    = t.ubVfull,
    dubVfull   = t.dubVfull,
    lse        = t.lse,
    tmp        = t.tmp,
    IminusTEVp = t.IminusTEVp,
    wp         = p.wp,
    Πz         = p.Πz,
    β          = p.β

    @testset "Gradient of EV for infill" begin
        θ1 = similar(θt)
        θ2 = similar(θt)
        fdEV = similar(evs.dEV)
        roy = 0.25
        geoid = 4.7
        itype = (geoid, roy,)

        for k = 1:length(θt)
            h = peturb(θt[k])
            θ1 .= θt
            θ2 .= θt
            θ1[k] -= h
            θ2[k] += h
            hh = θ2[k] - θ1[k]

            for i in ind_inf(wp)
                idxd, idxs, horzn = dp1space(wp,i), collect(sprimes(wp,i)), _horizon(wp,i)
                # idxd, idxs, horzn, st = wp_info(wp, i)

                @views ubV = ubVfull[:,:,idxd]
                @views dubV = dubVfull[:,:,:,idxd]
                @views EV0 = EV[:,:,i]
                @views dEV0 = dEV[:,:,:,i]

                fillflows!(ubV, flow, p, θt, σ, i, itype...)
                @views ubV .+= β .* EV[:,:,idxs]

                if dograd
                    fillflows_grad!(dubV, flowdθ, p, θt, σ, i, itype...)
                    fill!(dEV0, 0.0)
                    @views dubV .+= β .* dEV[:,:,:,idxs]
                end

                if dograd
                    vfit!(EV0, dEV0, ubV, dubV, lse, tmp, Πz)
                else
                    vfit!(EV0,       ubV,       lse, tmp, Πz)

            # vfit!(EV0, ubV, lse:AbstractMatrix, tmp::AbstractMatrix, Πz)
            # vfit!(EV0, dEV0, ubV, dubV::AbstractArray4, q::AbstractArray3, lse, tmp, Πz)

            solve_vf_terminal!(evs, prim)
            solve_vf_infill!(evs, tmpv, prim, θ1, σv, false, itype)
            fdEV[:,:,k,:] .= -evs.EV

            solve_vf_terminal!(evs, prim)
            solve_vf_infill!(evs, tmpv, prim, θ2, σv, false, itype)
            fdEV[:,:,k,:] .+= evs.EV
            fdEV[:,:,k,:] ./= hh
        end
        solve_vf_terminal!(evs, prim)
        solve_vf_infill!(evs, tmpv, prim, θt, σv, true, itype)

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
end
