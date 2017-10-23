let duex = similar(tmpv.duexσ), fduex = similar(tmpv.uex), roy=0.2, p=prim
    h = peturb(σv)
    σ1 = σv - h
    σ2 = σv + h
    hh = σ2 - σ1

    pdct = Base.product(zspace..., ψspace, 0:dmax(wp), 0:0, false )

    fillflows!(p.f,  duex, θt, σ1, pdct, roy)
    fillflows!(p.f, fduex, θt, σ2, pdct, roy)
    fduex .-= duex
    fduex ./= hh
    fillflows!(p.dfσ, duex, θt, σv, makepdct(p, θt, Val{:u}, σv), roy)

    @views maxv, idx = findmax(abs.(duex .- fduex))
    sub = ind2sub(duex, idx)
    @show "worst value is $maxv at $sub for duσ_add"
    @test duex ≈ fduex
end
