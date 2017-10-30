export u_add, udθ_add, udσ_add, udψ_add

θ0_uadd = [0.275755, -2.12373, -1.89473, -1.92316, -1.38247, -1.36998, -1.255, -1.28068, -1.28891, -1.29613, -1.28014, -2.11996, -0.818109, 1.85105, 0.566889]
θ1_uadd = [2.1556, 0.0296533, -0.515811, -0.881202, -0.784622, -0.445394, -0.578085, -0.541829, -0.444618, -0.738266, -1.09538, -3.79588, -2.68135, 1.23406, 0.207407]

function u_add(θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::Real, geoid::Real) where {T}
    d == 0 && return zero(T)
    u = exp(logp) * (one(T)-roy) * (Dgt0  ? θ[1]+ψ  :  θ[1]+ψ*_ρ2(σ) )  + (d==1 ?  θ[2] : θ[3])
    d>1      && (u *= convert(T,d))
    d1 == 1  && (u += θ[4])
    return u::T
end


function udθ_add(θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return  convert(T,d) * exp(logp) * (one(T)-roy)
    k == 2  && return  d  == 1 ? one(T)  : zero(T)
    k == 3  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 4  && return  d1 == 1 ? one(T)  : zero(T)
    throw(error("$k out of bounds"))
end

udσ_add(θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * ψ * -2.0 * σ / (one(T)+σ^2)^2
udψ_add(θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * exp(logp) * (one(T)-roy) * _ρ2(σ)

# ---------------------------------------------------------------------------

export u_addlin, udθ_addlin, udσ_addlin, udψ_addlin

function u_addlin(θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    u = exp(logp) * (one(T)-roy) * (Dgt0  ? θ[1] + θ[2]*geoid + θ[3]*ψ  :  θ[1] + θ[2]*geoid + θ[3]*ψ*_ρ2(σ) )  + (d==1 ?  θ[4] : θ[5])
    d>1      && (u *= d)
    d1 == 1  && (u += θ[6])
    return u::T
end


function udθ_addlin(θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return  convert(T,d) * exp(logp) * (one(T)-roy)
    k == 2  && return  convert(T,d) * exp(logp) * (one(T)-roy) * convert(T, geoid)
    k == 3  && return  convert(T,d) * exp(logp) * (one(T)-roy) * (Dgt0  ?  ψ  :  ψ*_ρ2(σ)  )

    k == 4  && return  d  == 1 ? one(T)  : zero(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 6  && return  d1 == 1 ? one(T)  : zero(T)

    throw(error("$k out of bounds"))
end

udσ_addlin(θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * θ[3] * exp(logp) * (one(T)-roy) * ψ * -2.0 * σ / (one(T)+σ^2)^2
udψ_addlin(θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * θ[3] * exp(logp) * (one(T)-roy) * _ρ2(σ)
