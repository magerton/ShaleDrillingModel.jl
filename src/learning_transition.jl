export _dΠψ!, _fdΠψ!, _Πψ!


# FIXME: Δ is a function of σ because step(ψspace) = 2*numsd*(1+σ^2)/(numpts-1).
#        This, however, means we have to comptue dψ/dσ ∀ ψ
# if false
#     _hΔ(σ::Real, nsd::Real, n::Int) = 0.5 * _ψstep(σ, nsd, n)
#     _dhΔdσ(σ::Real, nsd::Real, n::Int) = 0.5 * _dψstepdσv(σ, nsd, n)
#     _hΔ(σ::Real, nsd::Real, n::Int, h::Real) = _hΔ(σ+h, nsd, n)
# end

# levels versions
_ρ(σ::Real) = 1.0/sqrt(1.0+σ^2)
_ρ2(σ::Real) = 1.0/(1.0+σ^2)
_z(x2::Real, x1::Real, Δ::Real, ρ::Real) = (x2 - ρ^2*x1 + Δ)/sqrt(1.0-ρ)

# derivatives
_dzdρ(x2::Real, x1::Real, ρ::Real, z::Real) = -2.0*ρ*x1/sqrt(1.0-ρ) + 0.5*z/(1.0-ρ)
_dρdσ(σ::Real, ρ::Real) = -ρ^3 * σ
_dzdx1(ρ::Real) = - ρ^2 / sqrt(1.0-ρ)

# finite difference versions
_ρ(σ::Real, h::Real) = _ρ(σ+h)
_z(x2::Real, x1::Real, Δ::Real, ρ::Real, v::Real, h::Real) = _z(x2, x1+v*h, Δ, ρ)

function _dπdσ(x2::Real, x1::Real, Δ::Real, ρ::Real, σ::Real, v::Real)
    z = _z(x2,x1,Δ,ρ)
    return normpdf(z) * ( _dzdρ(x2,x1,ρ,z)*_dρdσ(σ,ρ) + _dzdx1(ρ)*v )
end

_dπdσ(x2::Real, x1::Real, Δ::Real, σ::Real, v::Real) = _dπdσ(x2,x1,Δ,_ρ(σ),σ,v)

# ------------------------------ matrix updates -------------------------

function _dβΠψ!(P::AbstractMatrix, y::StepRangeLen, σ::Real, β::Real, v::Real)
    n = length(y)
    (n,n) == size(P) || throw(DimensionMismatch())
    ρ = _ρ(σ)
    Δ = 0.5 * step(y)

    @inbounds for (j,yj) in enumerate(y)
        if j == 1
            @. P[ :,j] = β * _dπdσ(yj,y,Δ,ρ,σ,v)
        elseif j == n
            @. P[ :,j] = -β * _dπdσ(yj,y,-Δ,ρ,σ,v)
        else
            @. P[ :,j] = β * ( _dπdσ(yj,y,Δ,ρ,σ,v) - _dπdσ(yj,y,-Δ,ρ,σ,v) )
        end
    end
end



function _βΠψ!(P::AbstractMatrix, y::StepRangeLen, σ::Real, β::Real)
    n = length(y)
    (n,n) == size(P) || throw(DimensionMismatch())
    ρ = _ρ(σ)
    Δ = 0.5 * step(y)

    @inbounds for (j,yj) in enumerate(y)
        if j == 1
            @. P[ :,j] = β * normcdf(_z(yj, y, Δ, ρ))
        elseif j == n
            @. P[ :,j] = β * normccdf(_z(yj, y, -Δ, ρ))
        else
            @. P[ :,j] = β * ( normcdf(_z(yj, y, Δ, ρ)) - normcdf(_z(yj, y, -Δ, ρ) ))
        end
    end
end

function _fdβΠψ!(P::AbstractMatrix, y::StepRangeLen, σ::Real, β::Real, v::Real, h::Real)
    n = length(y)
    (n,n) == size(P) || throw(DimensionMismatch())
    ρ = _ρ(σ,h)
    Δ = 0.5 * step(y)

    @inbounds for (j,yj) in enumerate(y)
        if j == 1
            @. P[ :,j] = β * normcdf(_z(yj,y,Δ,ρ,v,h))
        elseif j == n
            @. P[ :,j] = β * normccdf(_z(yj,y,-Δ,ρ,v,h))
        else
            @. P[ :,j] = β * (normcdf(_z(yj,y,Δ,ρ,v,h)) - normcdf(_z(yj,y,-Δ,ρ,v,h)) )
        end
    end
end


# ------------------------------ derivative check -------------------------



function check_dπdσ(σ::T, ψspace::StepRangeLen, vspace::AbstractVector) where {T}
    h = max( abs(σ), one(T) ) * cbrt(eps(T))
    σp, σm = σ+h, σ-h
    hh = σp - σm
    Δ = 0.5 * step(ψspace)

    maxabs = 0.
    maxrel = 0.
    posabs = (Inf, Inf, Inf)
    posrel = (Inf, Inf, Inf)

    for ψj in ψspace
        for ψi in ψspace
            for v in vspace
                d = _dπdσ(ψj, ψi, Δ, σ, v)
                z2 = _z(ψj,ψi,Δ,_ρ(σ,h),v,h)
                z1 = _z(ψj,ψi,Δ,_ρ(σ,-h),v,-h)
                fd = (normcdf(z2) - normcdf(z1)) / hh
                absd = abs(d-fd)
                isfinite(d) || warn("non-finite dπdσ at ($ψj, $ψi, $v)")
                isfinite(fd) || warn("non-finite fd at ($ψj, $ψi, $v)")
                isapprox(d,fd, atol=1e-8) || warn("bad deriv: (ψj,ψi,v)=($ψj, $ψi, $v). fd = $fd, dπdσ = $d")
                reld = 2.0 * absd / (abs(d)+abs(fd))
                if absd > maxabs
                    maxabs = absd
                    posabs = (ψj, ψi, v)
                end
                if reld > maxrel && isfinite(reld)
                    maxrel = reld
                    posrel = (ψj, ψi, v)
                end
            end
        end
    end
    println("worst absdif = $maxabs at (ψj,ψi,v) = $posabs")
    println("worst reldif = $maxrel at (ψj,ψi,v) = $posrel")
    return (maxabs, maxrel)
end
