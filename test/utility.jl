
println("testing utility and flow gradients")

# check gradient of flow payoffs
check_flowgrad(θt, σv, prim, 0.2)




# ---------------- check that wrapper functions compute finite diffs correctly ------------------

let duexσ = similar(tmpv.duexσ),
    uex1 = similar(tmpv.uex),
    uex2 = similar(tmpv.uex),
    uin  = similar(tmpv.uin),
    v = vspace[1], h = cbrt(eps(Float64)), d1 = 0, Dgt0 = false, roy = 0.2,
    pdσ = makepdct(zspace, ψspace, vspace, wp, θt, Val{:duσ}),
    pdex = makepdct(zspace, ψspace, vspace, wp, θt, Val{:u})

    uin0 = @view(uin[:,:,:,1])
    uin1 = @view(uin[:,:,:,2])

    fillflows!(u_add, uin0, uin1, uex1, θt, σv, pdex, roy, v, -h)
    fillflows!(u_add, uin0, uin1, uex2, θt, σv, pdex, roy, v, h)
    fillflows!(duσ_add, duexσ, θt, σv, pdσ, roy)
    fduσex = (uex2 .- uex1) ./ (2.0*h)

    @test fduσex ≈ duexσ[:,:,1,:]
end


let duexσ = similar(tmpv.duexσ),
    uex1  = similar(tmpv.uex),
    uex2  = similar(tmpv.uex),
    fdex  = similar(tmpv.uex),
    vpos = 3,
    h = cbrt(eps(Float64)),
    roy = 0.2, p = prim, t = tmpv, σ = σv

    fillflows!(t, p, θt, σ, roy, vspace[vpos], -h)
    uex1 .= t.uex
    fillflows!(t, p, θt, σ, roy, vspace[vpos], h)
    uex2 .= t.uex
    fillflows!(p.dfσ, t.duexσ, θt, σ, makepdct(p, θt, Val{:duσ}, σ), roy)

    fdex .= (uex2 .- uex1) ./ (2.0.*h)
    vw = @view(t.duexσ[:,:,vpos,:])
    @test fdex ≈ vw
end
