@testset "σ gradient" begin

    let nψ = 31,
        nv = 31,
        wp = well_problem(dmx,4,10),
        ψspace = range(-6.0, stop=6.0, length=nψ),
        vspace = range(-3.0, stop=3.0, length=nv),
        z1space = (zspace[1],),
        prim = dcdp_primitives(:exproy, β, wp, z1space, Πp1, ψspace),
        tmpv = dcdp_tmpvars(prim),
        evs = dcdp_Emax(prim),
        σv = 0.5,
        itype = (0.2, 8),
        T = eltype(eltype(evs.EV)),
        h = peturb(σv),
        σ1 = σv - h,
        σ2 = σv + h,
        hh = σ2 - σ1,
        nSexp = _nSexp(wp)

        sev = SharedEV([1,], prim, [1.0/8.], 1:1)
        evs, typs = dcdp_Emax(sev, 1, 1)
        sitev = ItpSharedEV(sev, prim, σv)

        pdct = Base.product( z1space..., vspace, vspace, 1:nSexp)
        dEVσ = Array{T}(undef, size(pdct))
        EVσ1 = similar(dEVσ)
        EVσ2 = similar(dEVσ)
        fdEVσ = similar(dEVσ)
        itdEVσ = similar(dEVσ)

        @show size(tmpv.uin)
        @show size(tmpv.uex)
        @show length(tmpv.uex)
        @show length(ShaleDrillingModel.makepdct(prim, θt, Val{:u},  σv))

        solve_vf_all!(evs, tmpv, prim, θt, σv, itype, Val{true})
        # serial_prefilterByView!(sev,sitev)
        #
        # @show pdct
        #
        # for (i,xi) in enumerate(pdct)
        #     z, u, v, s = xi
        #     ψ = u + σv*v
        #     dpsi = gradient_d(Val{length(prim.zspace)+1}, sitev.EV, z, ψ, s, 1, 1)
        #     dEVσ[i] = dpsi*v + sitev.dEVσ[z,ψ,s,1,1]
        # end
        #
        # solve_vf_all!(evs, tmpv, prim, θt, σ1, itype, Val{false})
        # serial_prefilterByView!(sev,sitev,false)
        # for (i,xi) in enumerate(pdct)
        #     z, u, v, s = xi
        #     ψ = u + σ1*v
        #     EVσ1[i] = sitev.EV[z,ψ,s,1,1]
        # end
        #
        # solve_vf_all!(evs, tmpv, prim, θt, σ2, itype, Val{false})
        # serial_prefilterByView!(sev,sitev,false)
        # for (i,xi) in enumerate(pdct)
        #     z, u, v, s = xi
        #     ψ = u + σ2*v
        #     EVσ2[i] = sitev.EV[z,ψ,s,1,1]
        # end
        #
        # fdEVσ .= (EVσ2 .- EVσ1) ./ hh
        #
        # maxv, idx = findmax(abs.(fdEVσ .- dEVσ))
        # println("worst value is $maxv at $(CartesianIndices(fdEVσ)[idx]) for full dEV/dσ")
        # println("mean abs error is $(mean(abs.(fdEVσ .- dEVσ)))")
        # println("mean sqd error is $(var(fdEVσ .- dEVσ)). 90pctile = $(quantile(vec(abs.(fdEVσ .- dEVσ)), 0.9))")
        # @test fdEVσ ≈ dEVσ
    end
end
