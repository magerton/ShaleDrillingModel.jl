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

# functions in case we have regime-info
@inline flow(  FF::Type, θ::AbstractVector{T}, σ::T, logp::T, regime::Integer, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T} = flow(  FF, θ, σ, logp, ψ,    d, d1, Dgt0, roy, geoid)
@inline flowdθ(FF::Type, θ::AbstractVector{T}, σ::T, logp::T, regime::Integer, ψ::T, k::Integer, d::Integer, d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T} = flowdθ(FF, θ, σ, logp, ψ, k, d, d1, Dgt0, roy, geoid)
@inline flowdσ(FF::Type, θ::AbstractVector{T}, σ::T, logp::T, regime::Integer, ψ::T,             d::Integer,                          roy::T, geoid::Real) where {T} = flowdσ(FF, θ, σ, logp, ψ,    d,           roy, geoid)
@inline flowdψ(FF::Type, θ::AbstractVector{T}, σ::T, logp::T, regime::Integer, ψ::T,             d::Integer,                          roy::T, geoid::Real) where {T} = flowdψ(FF, θ, σ, logp, ψ,    d,           roy, geoid)

# -----------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- super simple: additive ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- additive, discrete geo ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- simple: additive ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- simple: exponential ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

@inline function expQ(θ1::T, θ2::T, θ3::T, σ::T, Dgt0::Bool, geoid::Real, ψ::Real) where {T<:Real}
    if Dgt0
        r = exp(θ1 + θ2*geoid + θ3*ψ)
    else
        ρ = _ρ(σ)
        r = exp(θ1 + θ2*geoid + θ3*ψ*ρ^2 + 0.5*(1.0-ρ) )
    end
    return r::T
end

@inline function flow(::Type{Val{:addexp}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    u = exp(logp) * (1.0-roy) * expQ(θ[1], θ[2], θ[3], σ, Dgt0, geoid, ψ) + (d==1 ?  θ[4] : θ[5])
    d>1      && (u *= d)
    d1 == 1  && (u += θ[6])
    return u::T
end

@inline function flowdθ(::Type{Val{:addexp}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return  convert(T,d) * exp(logp) * (one(T)-roy) * expQ(θ[1], θ[2], θ[3], σ, Dgt0, geoid, ψ)
    k == 2  && return  convert(T,d) * exp(logp) * (one(T)-roy) * expQ(θ[1], θ[2], θ[3], σ, Dgt0, geoid, ψ) * convert(T, geoid)
    k == 3  && return  convert(T,d) * exp(logp) * (one(T)-roy) * expQ(θ[1], θ[2], θ[3], σ, Dgt0, geoid, ψ) * (Dgt0 ? ψ : ψ * _ρ2(σ))

    k == 4  && return  d  == 1 ? one(T)  : zero(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 6  && return  d1 == 1 ? one(T)  : zero(T)

    throw(error("$k out of bounds"))
end

@inline function flowdσ(::Type{Val{:addexp}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    ρ = _ρ(σ)
    return convert(T,d) * exp(logp) * (one(T)-roy) * expQ(θ[1], θ[2], θ[3], σ, false, geoid, ψ) * (θ[3]*ψ*2.0*ρ - 0.5) * _dρdσ(σ, ρ)
end
@inline function flowdψ(::Type{Val{:addexp}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T}
    return d == 0 ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * expQ(θ[1], θ[2], θ[3], σ, false, geoid, ψ) * θ[3] * _ρ2(σ)
end

# -----------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- breaking exponential ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

@inline function flow(::Type{Val{:breakexp}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    if !Dgt0
        u = exp(logp) * (1.0-roy) * expQ(θ[1], θ[2], θ[3], σ, Dgt0, geoid, ψ) + (d==1 ?  θ[4] : θ[5])
    else
        u = exp(logp) * (1.0-roy) * expQ(θ[6], θ[7], θ[8], σ, Dgt0, geoid, ψ) + (d==1 ?  θ[9] : θ[10])
    end
    d>1      && (u *= d)
    d1 == 1  && (u += θ[11])
    return u::T
end

@inline function flowdθ(::Type{Val{:breakexp}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return Dgt0           ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * expQ(θ[1], θ[2], θ[3], σ, Dgt0, geoid, ψ)
    k == 2  && return Dgt0           ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * expQ(θ[1], θ[2], θ[3], σ, Dgt0, geoid, ψ) * convert(T, geoid)
    k == 3  && return Dgt0           ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * expQ(θ[1], θ[2], θ[3], σ, Dgt0, geoid, ψ) * ψ * _ρ2(σ)
    k == 4  && return Dgt0 || d >  1 ? zero(T) : one(T)
    k == 5  && return Dgt0 || d == 1 ? zero(T) : convert(T,d)


    k == 6  && return !Dgt0           ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * expQ(θ[6], θ[7], θ[8], σ, Dgt0, geoid, ψ)
    k == 7  && return !Dgt0           ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * expQ(θ[6], θ[7], θ[8], σ, Dgt0, geoid, ψ) * convert(T, geoid)
    k == 8  && return !Dgt0           ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * expQ(θ[6], θ[7], θ[8], σ, Dgt0, geoid, ψ) * ψ
    k == 9  && return !Dgt0 || d >  1 ? zero(T) : one(T)
    k == 10 && return !Dgt0 || d == 1 ? zero(T) : convert(T,d)

    k == 11  && return  d1 == 1 ? one(T)  : zero(T)

    throw(error("$k out of bounds"))
end

@inline flowdσ(::Type{Val{:breakexp}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = flowdσ(Val{:addexp}, θ, σ, logp, ψ, d, roy, geoid)
@inline flowdψ(::Type{Val{:breakexp}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = flowdψ(Val{:addexp}, θ, σ, logp, ψ, d, roy, geoid)


# -----------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- simple: addative w/ one break ---------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------


@inline function flow(::Type{Val{:linbreak}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    u = exp(logp) * (one(T)-roy) * (Dgt0  ? θ[1] + θ[2]*geoid + θ[3]*ψ  :  θ[1] + θ[2]*geoid + θ[4]*ψ*_ρ2(σ) )  + (d==1 ?  θ[5] : θ[6])
    d>1      && (u *= d)
    d1 == 1  && (u += θ[7])
    return u::T
end


@inline function flowdθ(::Type{Val{:linbreak}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return         convert(T,d) * exp(logp) * (one(T)-roy)
    k == 2  && return         convert(T,d) * exp(logp) * (one(T)-roy) * convert(T, geoid)
    k == 3  && return  Dgt0 ? convert(T,d) * exp(logp) * (one(T)-roy) *  ψ : zero(T)
    k == 4  && return  Dgt0 ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * ψ*_ρ2(σ)


    k == 5  && return  d  == 1 ? one(T)  : zero(T)
    k == 6  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 7  && return  d1 == 1 ? one(T)  : zero(T)

    throw(error("$k out of bounds"))
end

@inline flowdσ(::Type{Val{:linbreak}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * θ[4] * exp(logp) * (one(T)-roy) * ψ * -2.0 * σ / (one(T)+σ^2)^2
@inline flowdψ(::Type{Val{:linbreak}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * θ[4] * exp(logp) * (one(T)-roy) * _ρ2(σ)



# ---------------------------------------------------------------------------

@inline function flow(::Type{Val{:bigbreak}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    if !Dgt0
        u = exp(logp) * (one(T)-roy) * ( θ[1] + θ[2]*geoid + θ[3]*ψ *_ρ2(σ) ) + (d==1 ?  θ[4] : θ[5])
    else
        u = exp(logp) * (one(T)-roy) * ( θ[6] + θ[7]*geoid + θ[8]*ψ )         + (d==1 ?  θ[9] : θ[10])
    end
    d>1      && (u *= d)
    d1 == 1  && (u += θ[11])
    return u::T
end


@inline function flowdθ(::Type{Val{:bigbreak}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return  Dgt0           ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy)
    k == 2  && return  Dgt0           ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * convert(T, geoid)
    k == 3  && return  Dgt0           ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * ψ * _ρ2(σ)
    k == 4  && return  Dgt0 || d >  1 ? zero(T) : one(T)
    k == 5  && return  Dgt0 || d == 1 ? zero(T) : convert(T,d)

    k == 6  && return !Dgt0           ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy)
    k == 7  && return !Dgt0           ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * convert(T, geoid)
    k == 8  && return !Dgt0           ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * ψ
    k == 9  && return !Dgt0 || d >  1 ? zero(T) : one(T)
    k == 10 && return !Dgt0 || d == 1 ? zero(T) : convert(T,d)

    k == 11  && return  d1 == 1 ? one(T)  : zero(T)

    throw(error("$k out of bounds"))
end

@inline flowdσ(::Type{Val{:bigbreak}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * θ[3] * exp(logp) * (one(T)-roy) * ψ * -2.0 * σ / (one(T)+σ^2)^2
@inline flowdψ(::Type{Val{:bigbreak}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * θ[3] * exp(logp) * (one(T)-roy) * _ρ2(σ)


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
