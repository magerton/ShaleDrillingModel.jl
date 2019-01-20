# ---------------- logsumexp ------------------

@testset "testing logsumexp3" begin

    i = ShaleDrillingModel._nS(wp)-3
    dmaxp1 = ShaleDrillingModel._dmax(wp,i)+1

    tmpv = dcdp_tmpvars(prim)
    ubV  = @view(tmpv.ubVfull[ :,:,1:dmaxp1])
    dubV = @view(tmpv.dubVfull[:,:,:,1:dmaxp1])

    let geoid = 2,
        roy = 0.25,
        itype = (geoid, roy,)

        @views fillflows!(      ubV,   flow, prim, θt, σv, i, itype...)
        @views fillflows_grad!(dubV, flowdθ, prim, θt, σv, i, itype...)
    end

    tst = zeros(size(tmpv.lse))
    lsetest = zeros(size(tmpv.lse))
    @views qvw =  tmpv.q[:,:,1:dmaxp1]
    @views EV0  = evs.EV[:,:,i]
    @views dEV0 = evs.dEV[:,:,:,i]
    lse = tmpv.lse
    tmp = tmpv.tmp

    lse .= 0.0
    tmp .= 0.0
    @test all(lse.==0.0)
    @test all(tmp.==0.0)

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
