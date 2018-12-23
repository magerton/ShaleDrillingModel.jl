@testset "Utility" begin

   duex = similar(tmpv.duσ)
   fduex = similar(tmpv.u)
   roy = 0.2
   geoid = 2
   itype=(geoid, roy,)
   p = prim
   h = peturb(σv)
   σ1 = σv - h
   σ2 = σv + h
   hh = σ2 - σ1
   st = state(2,2,0,0)

   fillflows!(duex,  flow, p, θt, σ1, st, itype...)
   fillflows!(fduex, flow, p, θt, σ2, st, itype...)
   fduex .-= duex
   fduex ./= hh
   fillflows!(duex, flowdσ, p, θt, σv, st, itype...)

   @views maxv, idx = findmax(abs.(duex .- fduex))
   sub = CartesianIndices(duex)[idx]
   @show "worst value is $maxv at $sub for duσ_add"
   @test duex ≈ fduex
end
