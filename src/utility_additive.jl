export u_add, du_add, duσ_add

function u_add(θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    d == 0 && return zero(logp)
    u = exp(logp) * omroy * (Dgt0  ? θ[1]+ψ  :  θ[1]+ψ*_ρ2(σ) )  + (d==1 ?  θ[2] : θ[3])
    d>1      && (u *= d)
    d1 == 1  && (u += θ[4])
    return u
end


function du_add(θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, omroy::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return  d * exp(logp) * omroy
    k == 2  && return  d  == 1 ? one(T)  : zero(T)
    k == 3  && return  d  == 1 ? zero(T) : convert(T, d)
    k == 4  && return  d1 == 1 ? one(T)  : zero(T)
    throw(error("$k out of bounds"))
end


function duσ_add(θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, v::T,                d::Integer, omroy::Real) where {T}
    d == 0  &&  return zero(T)
    ρ2 = _ρ2(σ)
    return d * exp(logp) * omroy * ρ2 * (v - 2.0*ρ2*σ*ψ)
end
