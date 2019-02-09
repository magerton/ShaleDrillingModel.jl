export check_EVjac

# ------------------------------ check total grad ------------------------------

reldiff(x::T, y::T) where {T<:Real} = x+y == zero(T) ? zero(T) : convert(T,2) * abs(x-y) / (abs(x)+abs(y))
absdiff(x::T, y::T) where {T<:Real} = abs(x-y)

function check_EVjac(e::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector{T}, σ::T, roy::Real=0.2) where {T}

    θ1, θ2 = similar(θ), similar(θ)
    nk = length(θ)
    EVfd  = zeros(T, size(e.EV))
    dEV   = zeros(T, size(e.dEV))
    dEVσ = zeros(T, size(e.dEVσ))

    e.EV .= 0.0
    e.dEV .= 0.0
    e.dEVσ .= 0.0

    solve_vf_all!(e, t, p, θ, σ, roy, Val{true})

    dEV .= e.dEV
    dEVσ .= e.dEVσ

    all(isfinite.(dEV)) || throw(error("dEV not finite"))
    all(isfinite.(dEVσ)) || throw(error("dEVσ not finite"))

    for k in 1:nk
        θ1 .= θ
        θ2 .= θ
        h = max( abs(θ[k]), one(T) ) * cbrt(eps(T))
        θ1[k] -= h
        θ2[k] += h
        hh = θ2[k] - θ1[k]
        solve_vf_all!(e, t, p, θ2, σ, roy, Val{false})
        EVfd .= e.EV
        solve_vf_all!(e, t, p, θ1, σ, roy, Val{false})
        EVfd .-= e.EV
        EVfd ./= hh
        @views isapprox(EVfd, dEV[:,:,k,:], atol=1e-7)  ||  @warn "Bad grad for θ[$k]"
        @views absd = maximum( absdiff.(EVfd, dEV[:,:,k,:] ) )
        @views reld = maximum( reldiff.(EVfd, dEV[:,:,k,:] ) )
        # println("In dimension $k, abs diff is $absd. max rel diff is $reld")
    end


    h = max( abs(σ), one(T) ) * cbrt(eps(T))
    σp, σm = σ+h, σ-h
    hh = σp - σm
    vspace = _vspace(p)
    nv = length(vspace)
    for k in sample(1:nv, 3; replace=false)
        v = vspace[k]
        solve_vf_all!(e, t, p, θ, σ, roy, v, h)
        EVfd .= e.EV
        all(isfinite.(e.EV)) || throw(error("EV not finite for σ+h"))
        solve_vf_all!(e, t, p, θ, σ, roy, v, -h)
        all(isfinite.(e.EV)) || throw(error("EV not finite for σ-h"))
        EVfd .-= e.EV
        EVfd ./= hh
        fdvw = @view(EVfd[:,:,1:_nSexp(p)])
        dEVvw = @view(dEVσ[:,:,k,1:end-1])
        all(isfinite.(fdvw)) || throw(error("finite diff check not finite for v[$k] = v"))

        @views fdvw ≈ dEVvw  ||  @warn "Bad grad for  v[$k] = $v"
        @views absd = maximum( absdiff.(fdvw, dEVvw ) )
        @views reld = maximum( reldiff.(fdvw, dEVvw ) )
        # println("For v[$k] = $v, abs diff is $absd. max rel diff is $reld")
    end
end
