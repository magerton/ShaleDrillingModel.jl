export _dψ1dθρ, _ρ, _ψ2, _ψ1

@inline _ρ(θρ::Real) = logistic(θρ)
@inline _dρdθρ(θρ::Real) = (z = logistic(θρ); z*(1-z) )

@inline _ρ2(θρ::Real) = _ρ(θρ)^2
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
