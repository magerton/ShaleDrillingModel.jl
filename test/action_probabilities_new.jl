@testset "action probabilities" begin
    let royalty_rates = [0.25,],
        geology_types = [3.5,],
        nroy = length(royalty_rates),
        ngeo = length(geology_types),
        roy = royalty_rates[1],
        geoid = geology_types[1],
        itype = (geoid, roy,),
        pids = [1,],
        rngs = (zspace...,                                        vspace, vspace, 0:dmax(wp), 1:length(wp), geology_types, royalty_rates,),
        idxs = (2:2:length(zspace[1])-1, 2:3:length(zspace[2])-1, 1:5:51, 1:5:51, 0:dmax(wp), 1:length(wp), Base.OneTo(ngeo), Base.OneTo(nroy),),
        idx_sz = length.((θfull, idxs...)),
        prim = dcdp_primitives(flowfuncname, β, wp, zspace, Πp, ψspace),
        tmpv = dcdp_tmpvars(prim),
        sev = SharedEV(pids, prim, rngs[end-1:end]...),
        isev = ItpSharedEV(sev, prim, σv),
        T = eltype(θfull),
        tmp = Vector{T}(undef, dmax(wp)+1),
        θ1 = similar(θfull),
        θ2 = similar(θfull),
        θttmp = Vector{Float64}(undef, prim.nθt),
        grad   = zeros(T, idx_sz),
        fdgrad = zeros(T, idx_sz)

        println("Testing logP")

        dograd = true

        zero!(sev)
        CR = CartesianIndices(length.(idxs))


        let uv = (1.,1.), z = getindex.(zspace, 7), dp1 = 1, s_idx = 1, itypidx = (1, 1,), tmpgrad = Vector{Float64}(undef, idx_sz[1])
            θtvw = _θt(θfull, prim.nθt)
            σ0 = _σv(θfull)
            @test idx_sz[1] == length(θtvw)+1
            @test length(tmpgrad) == idx_sz[1]
            @test length(tmpgrad) == length(θtvw)+1
            logP!(tmpgrad, tmp, θtvw, σ0, prim, isev, uv, z, dp1, s_idx, itypidx, true)
        end
        # -----------------------

        println("solving all")

        dograd = true

        serial_solve_vf_all!(sev, tmpv, prim, θfull, Val{dograd}; maxit0=40, maxit1=20, vftol=1e-10)
        println("solved round 1. doing logP")
        grad .= 0.0
        for CI in CR
            zpi, zvi, ui, vi, di, si, gi, ri = CI.I
            zp , zv , u , v , d , s , g , r  = getindex.(rngs, CI.I)

            θtvw = _θt(θfull, prim)
            σ0 = _σv(θfull)

            if d ∈ ShaleDrillingModel._actionspace(prim.wp, s) # && !(s ∈ ShaleDrillingModel.ind_lrn(prim.wp.endpts))
                @views lp = logP!(grad[:,CI], tmp, θtvw, σ0, prim, isev,  (u,v), (zp,zv,), d, s, (gi, ri,), dograd)
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


            serial_solve_vf_all!(sev, tmpv, prim, θ1, Val{dograd}; maxit0=40, maxit1=20, vftol=1e-10)
            print("logp: θ[$k]-h\t")
            for CI in CR
                zpi, zvi, ui, vi, di, si, gi, ri = CI.I
                zp , zv , u , v , d , s , g , r  = getindex.(rngs, CI.I)
                θtvw1 = _θt(θ1, prim)
                if d ∈ ShaleDrillingModel._actionspace(prim.wp, s) # && !(s ∈ ShaleDrillingModel.ind_lrn(prim.wp.endpts))
                    fdgrad[k,CI] -= logP!(Vector{T}(), tmp, θtvw1, σ1, prim, isev, (u,v), (zp,zv,), d, s, (gi,ri), dograd)
                end
            end

            println("solving again for logp: θ[$k]+h....")
            serial_solve_vf_all!(sev, tmpv, prim, θ2, Val{dograd}; maxit0=40, maxit1=20, vftol=1e-10)
            println("updating fdgrad")
            for CI in CR
                zpi, zvi, ui, vi, di, si, gi, ri = CI.I
                zp , zv , u , v , d , s , g , r  = getindex.(rngs, CI.I)
                θtvw2 = _θt(θ2, prim)
                if d ∈ ShaleDrillingModel._actionspace(prim.wp, s) # && !(s ∈ ShaleDrillingModel.ind_lrn(prim.wp.endpts))
                    fdgrad[k,CI] += logP!(Vector{T}(), tmp, θtvw2, σ2, prim, isev, (u,v), (zp,zv,), d, s, (gi,ri,), dograd)
                    fdgrad[k,CI] /= hh
                end
            end
        end

        println("checking gradient")

        @test !all(fdgrad .== 0.0)
        @test !all(grad .== 0.0)
        @test all(isfinite.(fdgrad))
        @test all(isfinite.(grad))


        @test fdgrad ≈ grad
        maxv, idx =  findmax(abs.(fdgrad .- grad))
        println("worst value is $maxv at $(CartesianIndices(fdgrad)[idx]) for dlogP")
        @test maxv < 1.5e-7
    end
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

# end
