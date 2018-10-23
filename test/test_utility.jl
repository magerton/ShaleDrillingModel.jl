@testset "Utility" begin

   duex = similar(tmpv.duexσ)
   fduex = similar(tmpv.uex)
   roy = 0.2
   geoid = 2
   itype=(geoid, roy,)
   p = prim
   h = peturb(σv)
   σ1 = σv - h
   σ2 = σv + h
   hh = σ2 - σ1

   pdct = Base.product(zspace..., ψspace, 0:dmax(wp), 0:0, false, true)

   fillflows!(flow(p), flow,  duex, θt, σ1, pdct, itype...)
   fillflows!(flow(p), flow, fduex, θt, σ2, pdct, itype...)
   fduex .-= duex
   fduex ./= hh
   fillflows!(flow(p), flowdσ, duex, θt, σv, makepdct(p, θt, Val{:u}, σv), itype...)

   @views maxv, idx = findmax(abs.(duex .- fduex))
   sub = CartesianIndices(duex)[idx]
   @show "worst value is $maxv at $sub for duσ_add"
   @test duex ≈ fduex
end
