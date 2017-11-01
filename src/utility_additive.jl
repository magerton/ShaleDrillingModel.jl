export flow, flowdθ, flowdσ, flowdψ

"""
make primitives. Note: flow payoffs, gradient, and grad wrt `σ` must have the following structure:
```julia
f(  ::Type{Val{FF}}, θ::AbstractVector{T}, σ::T,   z... , ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, roy::Real, geoid::Real)
df( ::Type{Val{FF}}, θ::AbstractVector{T}, σ::T,   z... , ψ::T, k::Integer, d::Integer, d1::Integer, Dgt0::Bool, roy::Real, geoid::Real)
dfσ(::Type{Val{FF}}, θ::AbstractVector{T}, σ::T,   z... , ψ::T,             d::Integer,                          roy::Real, geoid::Real)
dfψ(::Type{Val{FF}}, θ::AbstractVector{T}, σ::T,   z... , ψ::T,             d::Integer,                          roy::Real, geoid::Real)
```
"""

@inline function flow(::Type{Val{:add}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::Real, geoid::Real) where {T}
    d == 0 && return zero(T)
    u = exp(logp) * (one(T)-roy) * (Dgt0  ? θ[1]+ψ  :  θ[1]+ψ*_ρ2(σ) )  + (d==1 ?  θ[2] : θ[3])
    d>1      && (u *= convert(T,d))
    d1 == 1  && (u += θ[4])
    return u::T
end


@inline function flowdθ(::Type{Val{:add}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return  convert(T,d) * exp(logp) * (one(T)-roy)
    k == 2  && return  d  == 1 ? one(T)  : zero(T)
    k == 3  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 4  && return  d1 == 1 ? one(T)  : zero(T)
    throw(error("$k out of bounds"))
end

@inline flowdσ(::Type{Val{:add}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * ψ * -2.0 * σ / (one(T)+σ^2)^2
@inline flowdψ(::Type{Val{:add}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * _ρ2(σ)


# ---------------------------------------------------------------------------

@inline function flow(::Type{Val{:adddisc}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::Real, geoid::Real) where {T}
    d == 0 && return zero(T)

    u = exp(logp) * (one(T)-roy) * (Dgt0  ? θ[1]+θ[2]*ψ  :  θ[1]+θ[2]*ψ*_ρ2(σ) )  + (d==1 ?  θ[3] : θ[4])
    d>1      && (u *= d)
    d1 == 1  && (u += θ[5])

    return T(u)
end


@inline function flowdθ(::Type{Val{:adddisc}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return  convert(T,d) * exp(logp) * (one(T)-roy)
    k == 2  && return  convert(T,d) * exp(logp) * (one(T)-roy) * (Dgt0 ? ψ : ψ * _ρ2(σ))
    k == 3  && return  d  == 1 ? one(T)  : zero(T)
    k == 4  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 5  && return  d1 == 1 ? one(T)  : zero(T)
    throw(error("$k out of bounds"))
end

@inline flowdσ(::Type{Val{:adddisc}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * θ[2] * ψ * -2.0 * σ / (one(T)+σ^2)^2
@inline flowdψ(::Type{Val{:adddisc}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * θ[2] * _ρ2(σ)

# ---------------------------------------------------------------------------

@inline function flow(::Type{Val{:addlin}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    u = exp(logp) * (one(T)-roy) * (Dgt0  ? θ[1] + θ[2]*geoid + θ[3]*ψ  :  θ[1] + θ[2]*geoid + θ[3]*ψ*_ρ2(σ) )  + (d==1 ?  θ[4] : θ[5])
    d>1      && (u *= d)
    d1 == 1  && (u += θ[6])
    return u::T
end


@inline function flowdθ(::Type{Val{:addlin}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return  convert(T,d) * exp(logp) * (one(T)-roy)
    k == 2  && return  convert(T,d) * exp(logp) * (one(T)-roy) * convert(T, geoid)
    k == 3  && return  convert(T,d) * exp(logp) * (one(T)-roy) * (Dgt0  ?  ψ  :  ψ*_ρ2(σ)  )

    k == 4  && return  d  == 1 ? one(T)  : zero(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 6  && return  d1 == 1 ? one(T)  : zero(T)

    throw(error("$k out of bounds"))
end

@inline flowdσ(::Type{Val{:addlin}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * θ[3] * exp(logp) * (one(T)-roy) * ψ * -2.0 * σ / (one(T)+σ^2)^2
@inline flowdψ(::Type{Val{:addlin}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * θ[3] * exp(logp) * (one(T)-roy) * _ρ2(σ)

# -------------------------------- cost heterogeneity -----------------------------------

# with prices only
@inline function flow(::Type{Val{:addlincost}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    u = exp(logp) * (one(T)-roy) * (Dgt0  ? θ[1] + θ[2]*geoid + θ[3]*ψ  :  θ[1] + θ[2]*geoid + θ[3]*ψ*_ρ2(σ) )  + (d==1 ?  θ[4] : θ[5])
    Dgt0     || (u += θ[6])
    d>1      && (u *= d)
    d1 == 1  && (u += θ[7])
    return u::T
end


@inline function flowdθ(::Type{Val{:addlincost}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return  convert(T,d) * exp(logp) * (one(T)-roy)
    k == 2  && return  convert(T,d) * exp(logp) * (one(T)-roy) * convert(T, geoid)
    k == 3  && return  convert(T,d) * exp(logp) * (one(T)-roy) * (Dgt0  ?  ψ  :  ψ*_ρ2(σ)  )

    k == 4  && return  d  == 1 ? one(T)  : zero(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 6  && return  Dgt0    ? zero(T) : convert(T,d)
    k == 7  && return  d1 == 1 ? one(T)  : zero(T)

    throw(error("$k out of bounds"))
end


@inline flowdσ(::Type{Val{:addlincost}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = flowdσ(Val{:addlin}, θ, σ, logp, ψ, d, roy, geoid)
@inline flowdψ(::Type{Val{:addlincost}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = flowdψ(Val{:addlin}, θ, σ, logp, ψ, d, roy, geoid)

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


# with regime switching + costs
@inline function flow(::Type{Val{:lintcost}}, θ::AbstractVector{T}, σ::T,    logp::T, logc::T, regime::Integer, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    u = exp(logp) * (one(T)-roy) * (Dgt0  ? θ[1] + θ[2]*geoid + θ[3]*ψ  :  θ[1] + θ[2]*geoid + θ[3]*ψ*_ρ2(σ) )  + θ[4]*exp(logc) + (d==1 ?  θ[5] : θ[6])
    Dgt0     || (u += θ[7])
    d>1      && (u *= d)
    d1 == 1  && (u += θ[8])
    return u::T
end


# with regime switching + costs
@inline function flowdθ(::Type{Val{:lintcost}}, θ::AbstractVector{T}, σ::T,     logp::T, logc::T, regime::Integer,   ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return  convert(T,d) * exp(logp) * (one(T)-roy)
    k == 2  && return  convert(T,d) * exp(logp) * (one(T)-roy) * convert(T, geoid)
    k == 3  && return  convert(T,d) * exp(logp) * (one(T)-roy) * (Dgt0  ?  ψ  :  ψ*_ρ2(σ)  )
    k == 4  && return  convert(T,d) * exp(logc)

    k == 5  && return  d  == 1 ? one(T)  : zero(T)
    k == 6  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 7  && return  Dgt0    ? zero(T) : convert(T,d)
    k == 8  && return  d1 == 1 ? one(T)  : zero(T)

    throw(error("$k out of bounds"))
end

@inline flowdσ(::Type{Val{:lintcost}}, θ::AbstractVector{T}, σ::T, logp::T, logc::T, regime::Integer, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = flowdσ(Val{:addlin}, θ, σ, logp, ψ, d, roy, geoid)
@inline flowdψ(::Type{Val{:lintcost}}, θ::AbstractVector{T}, σ::T, logp::T, logc::T, regime::Integer, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = flowdψ(Val{:addlin}, θ, σ, logp, ψ, d, roy, geoid)




# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


# with regime switching + costs
@inline function flow(::Type{Val{:linct}}, θ::AbstractVector{T}, σ::T,    logp::T, logc::T, regime::Integer, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    u = exp(logp) * (one(T)-roy) * (Dgt0  ? θ[1] + θ[2]*geoid + θ[3]*ψ  :  θ[1] + θ[2]*geoid + θ[3]*ψ*_ρ2(σ) )  + θ[4]*exp(logc) + (d==1 ?  θ[5] : θ[6])
    d>1      && (u *= d)
    d1 == 1  && (u += θ[7])
    return u::T
end


# with regime switching + costs
@inline function flowdθ(::Type{Val{:linct}}, θ::AbstractVector{T}, σ::T,     logp::T, logc::T, regime::Integer,   ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return  convert(T,d) * exp(logp) * (one(T)-roy)
    k == 2  && return  convert(T,d) * exp(logp) * (one(T)-roy) * convert(T, geoid)
    k == 3  && return  convert(T,d) * exp(logp) * (one(T)-roy) * (Dgt0  ?  ψ  :  ψ*_ρ2(σ)  )
    k == 4  && return  convert(T,d) * exp(logc)

    k == 5  && return  d  == 1 ? one(T)  : zero(T)
    k == 6  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 7  && return  d1 == 1 ? one(T)  : zero(T)

    throw(error("$k out of bounds"))
end

@inline flowdσ(::Type{Val{:linct}}, θ::AbstractVector{T}, σ::T, logp::T, logc::T, regime::Integer, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = flowdσ(Val{:addlin}, θ, σ, logp, ψ, d, roy, geoid)
@inline flowdψ(::Type{Val{:linct}}, θ::AbstractVector{T}, σ::T, logp::T, logc::T, regime::Integer, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = flowdψ(Val{:addlin}, θ, σ, logp, ψ, d, roy, geoid)
