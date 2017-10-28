
let dpi = similar(tmpv.Πψtmp), fdpi = similar(tmpv.Πψtmp)
    h = peturb(σv)
    σ1 = σv - h
    σ2 = σv + h
    hh = σ2 - σ1

    ShaleDrillingModel._βΠψ!(dpi, ψspace, σ1, β)
    ShaleDrillingModel._βΠψ!(fdpi, ψspace, σ2, β)
    fdpi .-= dpi
    fdpi ./= hh
    ShaleDrillingModel._βΠψdσ!(dpi, ψspace, σv, β)

    @views maxv, idx = findmax(abs.(dpi .- fdpi))
    sub = ind2sub(fdpi, idx)
    @show "worst value is $maxv at $sub for β dΠ/dσ"

    @test 0.0 < maxv < 1e-5
    @test fdpi ≈ dpi

    h = peturb(3.0)
    hh = 2.0*h

    ShaleDrillingModel._βΠψ!(dpi,  ψspace-h, ψspace, σv, β)
    ShaleDrillingModel._βΠψ!(fdpi, ψspace+h, ψspace, σv, β)
    fdpi .-= dpi
    fdpi ./= hh
    ShaleDrillingModel._βΠψdψ!(dpi, ψspace, σv, β)

    @views maxv, idx = findmax(abs.(dpi .- fdpi))
    sub = ind2sub(fdpi, idx)
    @show "worst value is $maxv at $sub for β dΠ/dψ"
    @test fdpi ≈ dpi
end
