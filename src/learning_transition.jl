export check_dΠψ


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
_z(x2::Real, x1::Real, Δ::Real, ρ::Real, h::Real) = _z(x2, x1+h, Δ, ρ)

function _dπdσ(x2::Real, x1::Real, Δ::Real, ρ::Real, σ::Real)
    z = _z(x2,x1,Δ,ρ)
    return normpdf(z) * _dzdρ(x2,x1,ρ,z)*_dρdσ(σ,ρ)
end

_dπdσ(x2::Real, x1::Real, Δ::Real, σ::Real) = _dπdσ(x2,x1,Δ,_ρ(σ),σ)

_dπdψ(x2::Real, x1::Real, Δ::Real, ρ::Real) = normpdf(_z(x2,x1,Δ,ρ)) * _dzdx1(ρ)

# ------------------------------ matrix updates -------------------------

function _βΠψdσ!(P::AbstractMatrix, y::StepRangeLen, σ::Real, β::Real)
    n = length(y)
    (n,n) == size(P) || throw(DimensionMismatch())
    ρ = _ρ(σ)
    Δ = 0.5 * step(y)

    @inbounds for (j,yj) in enumerate(y)
        if j == 1
            @. P[ :,j] = β * _dπdσ(yj,y,Δ,ρ,σ)
        elseif j == n
            @. P[ :,j] = -β * _dπdσ(yj,y,-Δ,ρ,σ)
        else
            @. P[ :,j] = β * ( _dπdσ(yj,y,Δ,ρ,σ) - _dπdσ(yj,y,-Δ,ρ,σ) )
        end
    end
end


function _βΠψdψ!(P::AbstractMatrix, y::StepRangeLen, σ::Real, β::Real)
    n = length(y)
    (n,n) == size(P) || throw(DimensionMismatch())
    ρ = _ρ(σ)
    Δ = 0.5 * step(y)

    @inbounds for (j,yj) in enumerate(y)
        if j == 1
            @. P[ :,j] = β * _dπdψ(yj,y,Δ,ρ)
        elseif j == n
            @. P[ :,j] = -β * _dπdψ(yj,y,-Δ,ρ)
        else
            @. P[ :,j] = β * ( _dπdψ(yj,y,Δ,ρ) - _dπdψ(yj,y,-Δ,ρ) )
        end
    end
end


function _βΠψ!(P::AbstractMatrix, y1::StepRangeLen, y2::StepRangeLen, σ::Real, β::Real)
    n = length(y2)
    n == length(y1) || throw(DimensionMismatch())
    (n,n) == size(P) || throw(DimensionMismatch())
    ρ = _ρ(σ)
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

_βΠψ!(P::AbstractMatrix, y1::StepRangeLen, σ::Real, β::Real) = _βΠψ!(P, y1, y1, σ, β)

# ------------------------------ derivative check -------------------------

function check_dΠψ(σ::Real, ψspace::StepRangeLen)

    Δ = 0.5 * step(ψspace)
    ρ = _ρ(σ)

    for yj in ψspace
        for y in ψspace
            fdσ = Calculus.derivative((sig) -> normcdf(_z(yj, y  , Δ, _ρ(sig))), σ)
            fdψ = Calculus.derivative((psi) -> normcdf(_z(yj, psi, Δ, ρ      )), y)
            dσ  = _dπdσ(yj, y, Δ, σ)
            dψ1 = _dπdψ(yj, y, Δ, ρ)
            abs(fdσ - dσ ) < 1e-7 || throw(error("bad σ grad at σ = $σ, ψ2 = $yj, ψ1 = $y"))
            abs(fdψ - dψ1) < 1e-7 || throw(error("bad ψ grad at σ = $σ, ψ2 = $yj, ψ1 = $y"))
        end
    end
    return true
end
