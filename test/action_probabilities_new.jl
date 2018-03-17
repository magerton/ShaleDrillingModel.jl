let pids = [1,],
    rngs = (zspace..., vspace, vspace, 0:dmax(wp), 1:length(wp), [0.2], 1:1),
    idxs = (1:2:31,    1:5:51, 1:5:51, 0:dmax(wp), 1:length(wp),  1:1 , 1:1),
    idx_sz = length.((θfull, idxs...)),
    sev = SharedEV(pids, prim, rngs[end-1:end]...),
    isev = ItpSharedEV(sev, prim, σv),
    T = eltype(θfull),
    tmp = Vector{T}(dmax(wp)+1),
    θ1 = similar(θfull),
    θ2 = similar(θfull),
    θttmp = Vector{Float64}(prim.nθt),
    grad   = zeros(T, idx_sz),
    fdgrad = zeros(T, idx_sz)

    println("Testing logP & gradient")

    dograd = true

    zero!(sev)
    CR = CartesianRange(length.(idxs))

    let uv = (1.,1.), z = (1.2,), dp1 = 1, s = 1, itypidx = (1, 1,)
        θtvw = _θt(θfull, prim.nθt)
        σ0 = _σv(θfull)
        logP!(Vector{Float64}(idx_sz[1]), tmp, θtvw, σ0, prim, isev, uv, z, dp1, s, itypidx, true)
    end
    # -----------------------


    dograd = true
    serial_solve_vf_all!(sev, tmpv, prim, θfull, Val{dograd}; maxit0=12, maxit1=20, vftol=1e-10)
    println("solved round 1. doing logP")
    grad .= 0.0
    for CI in CR
        zi, ui, vi, di, si, ri, gi = CI.I
        z, u, v, d, s, r, g = getindex.(rngs, CI.I)

        θtvw = _θt(θfull, prim)
        σ0 = _σv(θfull)

        if d ∈ ShaleDrillingModel._actionspace(prim.wp, s) # && !(s ∈ ShaleDrillingModel.ind_lrn(prim.wp.endpts))
            @views lp = logP!(grad[:,CI], tmp, θtvw, σ0, prim, isev,  (u,v), (z,), d, s, (ri, gi,), dograd)
        end
    end

    dograd = false
    for k in 1:length(θfull)
        print("θfull[$k]...")
        θ1 .= θfull
        θ2 .= θfull
        h = peturb(θfull[k])
        θ1[k] -= h
        θ2[k] += h
        hh = θ2[k] - θ1[k]

        σ1 = _σv(θ1)
        σ2 = _σv(θ2)


        serial_solve_vf_all!(sev, tmpv, prim, θ1, Val{dograd}; maxit0=12, maxit1=20, vftol=1e-10)
        print("logp: θ[$k]-h")
        for CI in CR
            zi, ui, vi, di, si, ri, gi = CI.I
            z, u, v, d, s, r, g = getindex.(rngs, CI.I)
            θtvw1 = _θt(θ1, prim)
            if d ∈ ShaleDrillingModel._actionspace(prim.wp, s) # && !(s ∈ ShaleDrillingModel.ind_lrn(prim.wp.endpts))
                fdgrad[k,CI] -= logP!(Vector{T}(0), tmp, θtvw1, σ1, prim, isev, (u,v), (z,), d, s, (ri,gi), dograd)
            end
        end

        println("solving again for logp: θ[$k]+h....")
        serial_solve_vf_all!(sev, tmpv, prim, θ2, Val{dograd}; maxit0=12, maxit1=20, vftol=1e-10)
        print("updating fdgrad\n")
        for CI in CR
            zi, ui, vi, di, si, ri, gi = CI.I
            z, u, v, d, s, r, g = getindex.(rngs, CI.I)
            θtvw2 = _θt(θ2, prim)
            if d ∈ ShaleDrillingModel._actionspace(prim.wp, s) # && !(s ∈ ShaleDrillingModel.ind_lrn(prim.wp.endpts))
                fdgrad[k,CI] += logP!(Vector{T}(0), tmp, θtvw2, σ2, prim, isev, (u,v), (z,), d, s, (ri,gi), dograd)
                fdgrad[k,CI] /= hh
            end
        end
    end

    println("checking gradient")

    @test !all(fdgrad .== 0.0)
    @test !all(grad .== 0.0)
    @test all(isfinite.(fdgrad))
    @test all(isfinite.(grad))


    maxv, idx =  findmax(abs.(fdgrad .- grad))
    println("worst value is $maxv at $(ind2sub(fdgrad, idx)) for dlogP")
    @test fdgrad ≈ grad
    @test maxv < 1.5e-7
end






# println("With σ")
# maxv, idx =  findmax(abs.(fdgrad_vw1 .- grad_vw1))
# mae = mean(abs.(fdgrad_vw1 .- grad_vw1))
# mse = var(fdgrad_vw1 .- grad_vw1)
# med = median(abs.(fdgrad_vw1 .- grad_vw1))
# q90 = quantile(vec(abs.(fdgrad_vw1 .- grad_vw1)), 0.9)
# sub = ind2sub(grad_vw1, idx)
# vals = getindex.(rngs, sub)
# println("worst value is $maxv at $sub for dlogP. This has characteristics $vals")
# println("MAE = $mae. MSE = $mse. Median abs error = $med. 90pctile = $q90")
# @test 0.0 < maxv < 0.1
