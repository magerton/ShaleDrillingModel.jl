export u_add, udθ_add, udσ_add, udψ_add

function u_add(θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, omroy::T) where {T}
    d == 0 && return zero(T)
    u = exp(logp) * omroy * (Dgt0  ? θ[1]+ψ  :  θ[1]+ψ*_ρ2(σ) )  + (d==1 ?  θ[2] : θ[3])
    d>1      && (u *= convert(T,d))
    d1 == 1  && (u += θ[4])
    return u::T
end


function udθ_add(θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, omroy::T) where {T}
    d == 0  && return zero(T)

    k == 1  && return  d * exp(logp) * omroy
    k == 2  && return  d  == 1 ? one(T)  : zero(T)
    k == 3  && return  d  == 1 ? zero(T) : convert(T, d)
    k == 4  && return  d1 == 1 ? one(T)  : zero(T)
    throw(error("$k out of bounds"))
end

udσ_add(θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, omroy::T) where {T} = d == 0 ? zero(T) : convert(T,d) * exp(logp) * omroy * ψ * -2.0 * σ / (one(T)+σ^2)^2
udψ_add(θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, omroy::T) where {T} = d == 0 ? zero(T) : convert(T,d) * exp(logp) * omroy * _ρ2(σ)
