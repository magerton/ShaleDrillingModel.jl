export check_dΠψ, _dψ1dθρ, _ρ, _ψ2, _ψ1


# FIXME: Δ is a function of σ because step(ψspace) = 2*numsd*(1+σ^2)/(numpts-1).
#        This, however, means we have to comptue dψ/dσ ∀ ψ
# if false
#     _hΔ(σ::Real, nsd::Real, n::Int) = 0.5 * _ψstep(σ, nsd, n)
#     _dhΔdσ(σ::Real, nsd::Real, n::Int) = 0.5 * _dψstepdσv(σ, nsd, n)
#     _hΔ(σ::Real, nsd::Real, n::Int, h::Real) = _hΔ(σ+h, nsd, n)
# end

# levels versions
@inline _ρ(θρ::Real) = 2 * logistic(θρ) - 1

@inline _ρ2(θρ::Real) = _ρ(θρ)^2

@inline _dρdθρ(θρ::Real) = (ex = exp(-θρ); 2 * ex / (1+ex)^2)

_dρdσ = _dρdθρ # alias

@inline _ψ1(u::Real,v::Real,ρ::Real) = ρ*u + sqrt(1-ρ^2)*v
@inline _ψ2(u::Real,v::Real, ρ::Real) = _ψ2(u,v)
@inline _ψ2(u::Real,v::Real) = u

@inline _dψ1dρ(u::Real,v::Real,ρ::Real) = u - ρ/sqrt(1-ρ^2)*v

@inline _dψ1dθρ(u::Real, v::Real, ρ::Real, θρ::Real) = _dψ1dρ(u,v,ρ)*_dρdθρ(θρ::Real)

@inline _z(x2::Real, x1::Real, Δ::Real, ρ::Real) = (x2 - ρ*x1 + Δ)/sqrt(1-ρ^2)

# derivatives
@inline _dzdρ(x2::Real, x1::Real, ρ::Real, z::Real) = -x1/sqrt(1-ρ^2) + ρ*z/(1-ρ^2)
@inline _dπdρ(x2::Real, x1::Real, Δ::Real, ρ::Real) = (z = _z(x2,x1,Δ,ρ);  normpdf(z) * _dzdρ(x2,x1,ρ,z))

# finite difference versions
@inline _ρ(x::Real, h::Real) = _ρ(x+h)
@inline _z(x2::Real, x1::Real, Δ::Real, ρ::Real, h::Real) = _z(x2, x1+h, Δ, ρ)

# ------------------------------ matrix updates -------------------------

function _βΠψdθρ!(P::AbstractMatrix, y::StepRangeLen, θρ::Real, β::Real)
    n = length(y)
    (n,n) == size(P) || throw(DimensionMismatch())
    ρ = _ρ(θρ)
    dρdθρ = _dρdθρ(θρ)
    Δ = 0.5 * step(y)

    @inbounds for (j,yj) in enumerate(y)
        if j == 1
            @. P[ :,j] = β * _dπdρ(yj,y,Δ,ρ) * dρdθρ
        elseif j == n
            @. P[ :,j] = -β * _dπdρ(yj,y,-Δ,ρ) * dρdθρ
        else
            @. P[ :,j] = β * ( _dπdρ(yj,y,Δ,ρ) - _dπdρ(yj,y,-Δ,ρ) ) * dρdθρ
        end
    end
end

function _βΠψ!(P::AbstractMatrix, y1::StepRangeLen, y2::StepRangeLen, θp::Real, β::Real)
    n = length(y2)
    n == length(y1) || throw(DimensionMismatch())
    (n,n) == size(P) || throw(DimensionMismatch())
    ρ = _ρ(θp)
    Δ = 0.5 * step(y2)

    @inbounds for (j,yj) in enumerate(y2)
        if j == 1
            @. P[ :,j] = β * normcdf(_z(yj, y1, Δ, ρ))
        elseif j == n
            @. P[ :,j] = β * normccdf(_z(yj, y1, -Δ, ρ))
        else
            @. P[ :,j] = β * ( normcdf(_z(yj, y1, Δ, ρ)) - normcdf(_z(yj, y1, -Δ, ρ) ))
        end
    end
end

_βΠψ!(P::AbstractMatrix, y1::StepRangeLen, θp::Real, β::Real) = _βΠψ!(P, y1, y1, θp, β)

function _βΠψ(y1::StepRangeLen{T}, θp::Real, β::Real) where {T}
    P = Matrix{T}(undef, length(y1), length(y1))
    _βΠψ!(P, y1, y1, θp, β)
    return P
end

# ------------------------------ derivative check -------------------------

function check_dΠψ(θρ::Real, ψspace::StepRangeLen)

    Δ = 0.5 * step(ψspace)

    for yj in ψspace
        for y in ψspace
            fdσ = Calculus.derivative((sig) -> normcdf(_z(yj, y, Δ, _ρ(sig))), θρ)
            dσ  = _dρdθρ(θρ) * _dπdρ(yj, y, Δ, _ρ(θρ))
            abs(fdσ - dσ ) < 1e-7 || throw(error("bad σ grad at σ = $σ, ψ2 = $yj, ψ1 = $y"))
        end
    end
    return true
end
