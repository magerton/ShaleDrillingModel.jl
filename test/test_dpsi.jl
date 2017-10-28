println("doing big test of ψ gradient")
let nψ = 101,
    nv = 41,
    wp = well_problem(dmx,4,10),
    ψspace = linspace(-6.0, 6.0, nψ),
    vspace = linspace(-3.0, 3.0, nv),
    prim = dcdp_primitives(u_addlin, udθ_addlin, udσ_addlin, udψ_addlin, β, wp, zspace, Πp1, ψspace, ngeo, length(θt)),
    tmpv = dcdp_tmpvars(prim),
    evs = dcdp_Emax(prim),
    σv = 0.9,
    itype = (0.2, 8),
    T = eltype(eltype(evs.EV)),
    h = peturb(σv),
    σ1 = σv - h,
    σ2 = σv + h,
    hh = σ2 - σ1,
    nSexp = _nSexp(wp)

    sev = SharedEV([1,], prim)  # only doing this for one type, so no type indices
    sitev = ItpSharedEV(sev, prim, σv)
    evs, typs = dcdp_Emax(sev)

    pdct = Base.product( zspace..., ψspace, 1:nSexp)
    dEVσ  = Array{T}(size(pdct))
    EVσ1  = 0.0 .* similar(dEVσ)
    EVσ2  = 0.0 .* similar(dEVσ)
    fdEVσ = 0.0 .* similar(dEVσ)
    dψ    = 0.0 .* similar(dEVσ)

    ShaleDrillingModel.check_size(prim, tmpv)
    ShaleDrillingModel.check_size(prim, evs)

    solve_vf_all!(evs, tmpv, prim, θt, σv, itype, Val{true})
    for (i,xi) in enumerate(pdct)
        # z, u, v, s = xi
        # ψ = u + σv*v
        z, ψ, s = xi
        dEVσ[i] = sitev.dEVψ[z,ψ,s] # sitev.dEVσ[z,ψ,s] + v *
        dψ[i] = gradient(sitev.EV, z, ψ, s)[2]
    end

    # zψs = pspace[31], ψspace[783], 11
    # @show (gradient(sitev.EV, zψs...) - sitev.dEVψ[zψs...])[2]

    solve_vf_all!(evs, tmpv, prim, θt, σv, itype, Val{false})
    for (i,xi) in enumerate(pdct)
        # z, u, v, s = xi
        # ψ = u + σv*v - h
        z, ψ, s = xi
        ψ -= h
        EVσ1[i] = sitev.EV[z,ψ,s,]
    end

    solve_vf_all!(evs, tmpv, prim, θt, σv, itype, Val{false})
    for (i,xi) in enumerate(pdct)
        # z, u, v, s = xi
        # ψ = u + σv*v + h
        z, ψ, s = xi
        ψ += h
        EVσ2[i] = sitev.EV[z,ψ,s,]
    end

    maxv_itp, idx = findmax(abs.(dψ .- dEVσ))
    sub = ind2sub(dψ, idx)
    @show "worst itp error is $maxv_itp at $sub"

    fdEVσ .= (EVσ2 .- EVσ1) ./ hh
    @views maxv, idx = findmax(abs.(fdEVσ .- dEVσ))
    sub = ind2sub(fdEVσ, idx)
    @show "worst value is $maxv at $sub for dEV/dψ"
    @test 0.0 < maxv < 1.e-2
    @test maxv < maxv_itp*1.1
end




println("doing big test of σ gradient")
let nψ = 31,
    nv = 31,
    wp = well_problem(dmx,4,10),
    ψspace = linspace(-6.0, 6.0, nψ),
    vspace = linspace(-3.0, 3.0, nv),
    prim = dcdp_primitives(u_addlin, udθ_addlin, udσ_addlin, udψ_addlin, β, wp, zspace, Πp1, ψspace, ngeo, length(θt)),
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

    sev = SharedEV([1,], prim)
    evs, typs = dcdp_Emax(sev)

    pdct = Base.product( zspace..., vspace, vspace, 1:nSexp)
    dEVσ = Array{T}(size(pdct))
    EVσ1 = similar(dEVσ)
    EVσ2 = similar(dEVσ)
    fdEVσ = similar(dEVσ)
    itdEVσ = similar(dEVσ)


    solve_vf_all!(evs, tmpv, prim, θt, σv, itype, Val{true})
    sitev = ItpSharedEV(sev, prim, σv, Quadratic(InPlace()))
    for (i,xi) in enumerate(pdct)
        z, u, v, s = xi
        ψ = u + σv*v
        dpsi_it = gradient(sitev.EV, z, ψ, s)[2]
        dpsi_an = sitev.dEVψ[z,ψ,s]
        dEVσ[i] = dpsi_an*v + sitev.dEVσ[z,ψ,s]
        itdEVσ[i] = dpsi_it*v + sitev.dEVσ[z,ψ,s]
    end

    solve_vf_all!(evs, tmpv, prim, θt, σ1, itype, Val{false})
    sitev = ItpSharedEV(sev, prim, σv, Quadratic(InPlace()))
    for (i,xi) in enumerate(pdct)
        z, u, v, s = xi
        ψ = u + σ1*v
        EVσ1[i] = sitev.EV[z,ψ,s]
    end

    solve_vf_all!(evs, tmpv, prim, θt, σ2, itype, Val{false})
    sitev = ItpSharedEV(sev, prim, σv, Quadratic(InPlace()))
    for (i,xi) in enumerate(pdct)
        z, u, v, s = xi
        ψ = u + σ2*v
        EVσ2[i] = sitev.EV[z,ψ,s]
    end

    fdEVσ .= (EVσ2 .- EVσ1) ./ hh


    maxv, idx = findmax(abs.(fdEVσ .- dEVσ))
    avgerr = mean(abs.(fdEVσ .- dEVσ))
    mse = var(fdEVσ .- dEVσ)
    q90 = quantile(vec(abs.(fdEVσ .- dEVσ)), 0.9)
    sub = ind2sub(fdEVσ, idx)
    println("Versus ANALYTIC")
    println("worst value is $maxv at $sub for full dEV/dσ")
    println("mean abs error is $avgerr")
    println("mean sqd error is $mse. 90pctile = $q90")

    maxv, idx = findmax(abs.(fdEVσ .- itdEVσ))
    avgerr = mean(abs.(fdEVσ .- itdEVσ))
    mse = var(fdEVσ .- itdEVσ)
    q90 = quantile(vec(abs.(fdEVσ .- itdEVσ)), 0.9)
    sub = ind2sub(fdEVσ, idx)
    println("Versus INTERPOLLATED")
    println("worst value is $maxv at $sub for full dEV/dσ")
    println("mean abs error is $avgerr")
    println("mean sqd error is $mse. 90pctile = $q90")
    # @show maxv_itp = maximum(abs.(dψerr))
    # @show error_over_itperr = maximum(abs.(fdEVσ .- dEVσ) .- abs.(dψerr))
    # @test error_over_itperr < 0.03
    # @test 0.0 < maxv < 4.0e-2
    # @test maxv < maxv_itp*2
end


















#
