@testset "Utility" begin

   duex = similar(tmpv.dubV_σ)
   fduex = similar(tmpv.ubVfull)
   roy = 0.2
   geoid = 2
   itype=(geoid, roy,)
   p = prim
   h = peturb(σv)
   σ1 = σv - h
   σ2 = σv + h
   hh = σ2 - σ1
   st = state(2,2,0,0)
   stidx = state_idx(p.wp,2,2,0,0)

   fillflows!(duex,  flow, p, θt, σ1, stidx, itype...)
   fillflows!(fduex, flow, p, θt, σ2, stidx, itype...)
   fduex .-= duex
   fduex ./= hh
   fillflows!(duex, flowdσ, p, θt, σv, stidx, itype...)

   @views maxv, idx = findmax(abs.(duex .- fduex))
   sub = CartesianIndices(duex)[idx]
   @show "worst value is $maxv at $sub for duσ_add"
   @test duex ≈ fduex
end
