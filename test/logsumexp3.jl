# ---------------- logsumexp ------------------

println("testing logsumexp3")



# test logsumexp
let i = 17,
    dmaxp1 = ShaleDrillingModel._nd(prim),
    tst = similar(tmpv.lse),
    lsetest = similar(tmpv.lse),
    qvw = @view(tmpv.q[:,:,1:dmaxp1]),
    EV0  = @view(evs.EV[:,:,i]),
    dEV0 = @view(evs.dEV[:,:,:,i]),
    ubV  = @view(tmpv.ubVfull[:,:,1:dmaxp1]),
    dubV = @view(tmpv.dubVfull[:,:,:,1:dmaxp1]),
    lse = tmpv.lse,
    tmp = tmpv.tmp

    ubV .= @view(tmpv.uin[:,:,1:dmaxp1,2])
    dubV .= @view(tmpv.duin[:,:,:,1:dmaxp1,2])

    logsumexp3!(lse,tmp,ubV)
    for i in 1:31, j in 1:5
        lsetest[i,j] = logsumexp(@view(ubV[i,j,:]))
    end
    @test all(lsetest .== lse)

    # logsumexp_and_softmax3
    logsumexp_and_softmax3!(lse, qvw, tmp, ubV)
    @test all(sum(qvw,3) .≈ 1.0)
    @test all(lsetest .== lse)

    # logsumexp_and_softmax3 - q0
    logsumexp_and_softmax3!(lse, tst, tmp, ubV)
    @test all(tst .== qvw[:,:,1])
    @test all(lsetest .== lse)
end
