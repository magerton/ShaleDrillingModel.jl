# ---------------- logsumexp ------------------

@testset "testing logsumexp3" begin

    i = 17

    let geoid = 2,
        roy = 0.25,
        itype = (geoid, roy,),
        st = state(wp,i)

        fillflows!(     tmpv.u,  flow,   prim, θt, σv, st, itype...)
        fillflows_grad!(tmpv.du, flowdθ, prim, θt, σv, st, itype...)
    end

    dmaxp1 = ShaleDrillingModel._nd(prim)
    tst = zeros(size(tmpv.lse))
    lsetest = zeros(size(tmpv.lse))
    qvw = @view(tmpv.q[:,:,1:dmaxp1])
    EV0  = @view(evs.EV[:,:,i])
    dEV0 = @view(evs.dEV[:,:,:,i])
    ubV  = @view(tmpv.ubVfull[:,:,1:dmaxp1])
    dubV = @view(tmpv.dubVfull[:,:,:,1:dmaxp1])
    lse = tmpv.lse
    tmp = tmpv.tmp

    lse .= 0.0
    tmp .= 0.0
    @test all(lse.==0.0)
    @test all(tmp.==0.0)

    @views  ubV .= tmpv.u[:,:,1:dmaxp1]
    @views dubV .= tmpv.du[:,:,:,1:dmaxp1]

    ShaleDrillingModel.logsumexp3!(lse,tmp,ubV)
    for CI in CartesianIndices(size(lsetest))
        @views lsetest[CI] = logsumexp(ubV[CI,:])
    end
    @show findmax(abs.(lsetest .- lse))
    @test lsetest ≈ lse

    # logsumexp_and_softmax3
    logsumexp_and_softmax3!(lse, qvw, tmp, ubV)
    @test all(sum(qvw, dims=3) .≈ 1.0)
    @show findmax(abs.(lsetest .- lse))
    @test lsetest ≈ lse

    # logsumexp_and_softmax3 - q0
    logsumexp_and_softmax3!(lse, tst, tmp, ubV)
    @test all(tst .== qvw[:,:,1])
    @test all(lsetest .== lse)
end
