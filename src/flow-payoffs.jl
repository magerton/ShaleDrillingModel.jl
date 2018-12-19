export flow, flowdθ, flowdσ, flowdψ, STARTING_σ_ψ, STARTING_log_ogip


# functions in case we have volatility regime
# @inline flowrev(FF::Type, θ::AbstractVector{T}, σ::T, logp::T,          logvol::Real, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool,                geoid::Real, roy::T) where {T} = flowrev(FF, θ, σ, logp,       ψ,    d, d1, Dgt0,                  geoid, roy)
# @inline flow(   FF::Type, θ::AbstractVector{T}, σ::T, logp::T,          logvol::Real, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {T} = flow(   FF, θ, σ, logp,       ψ,    d, d1, Dgt0, sgn_ext,         geoid, roy)
# @inline flowdθ( FF::Type, θ::AbstractVector{T}, σ::T, logp::T,          logvol::Real, ψ::T, k::Integer, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {T} = flowdθ( FF, θ, σ, logp,       ψ, k, d, d1, Dgt0, sgn_ext,         geoid, roy)
#
# @inline flowdψ( FF::Type, θ::AbstractVector{T}, σ::T, logp::T,           logvol::Real, ψ::T,             d::Integer, st::Union{state,Bool},                  geoid::Real, roy::T) where {T} = flowdψ( FF, θ, σ, logp, ψ,    d, st,                        geoid, roy)
# @inline flowdψ( FF::Type, θ::AbstractVector{T}, σ::T, logp::T, logc::T,  logvol::Real, ψ::T,             d::Integer, st::Union{state,Bool},                  geoid::Real, roy::T) where {T} = flowdψ( FF, θ, σ, logp, ψ,    d, st,                        geoid, roy)
# @inline flowdψ( FF::Type, θ::AbstractVector{T}, σ::T, logp::T,                         ψ::T,             d::Integer, st::state,                              geoid::Real, roy::T) where {T} = flowdψ( FF, θ, σ, logp, ψ,    d, _sign_lease_extension(st), geoid, roy)
# @inline flowdψ( FF::Type, θ::AbstractVector{T}, σ::T, logp::T, logc::T,                ψ::T,             d::Integer, st::state,                              geoid::Real, roy::T) where {T} = flowdψ( FF, θ, σ, logp, ψ,    d, _sign_lease_extension(st), geoid, roy)
#
# @inline flowdσ( FF::Type, θ::AbstractVector{T}, σ::T, logp::T,          logvol::Real, ψ::T, d::Integer, geoid::Real, roy::T) where {T} = flowdσ( FF, θ, σ, logp, ψ, d, geoid, roy)
# @inline flowdσ( FF::Type, θ::AbstractVector{T}, σ::T, logp::T, logc::T, logvol::Real, ψ::T, d::Integer, geoid::Real, roy::T) where {T} = flowdσ( FF, θ, σ, logp, ψ, d, geoid, roy)

# --------------------------- common revenue functions & derivatives  --------------------------------------

function Eexpψ(θ4::T, σ::T, ψ::T, Dgt0::Bool) where {T<:Real}
    if Dgt0
        out = θ4*ψ
    else
        ρ = _ρ(σ)
        out = θ4*(ψ*ρ + θ4*0.5*(one(T)-ρ^2))
    end
    return out::T
end

@inline function rev_exp(θ0::Real, θ1::T, θ2::Real, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real, Dgt0::Bool, geoid::Real, roy::Real) where {T<:Real}
    r = (one(T)-θ0*roy) * exp(θ1 + θ2*logp + θ3*geoid + Eexpψ(θ4, σ, ψ, Dgt0))
    return r::T
end

@inline function drevdσ_exp(θ0::Real, θ1::T, θ2::Real, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real, geoid::Real, roy::Real) where {T}
    return rev_exp(θ0,θ1,θ2,θ3,θ4,σ,logp,ψ,false,geoid, roy) * (ψ*θ4 - θ4^2*_ρ(σ)) * _dρdσ(σ)
end

@inline function drevdψ_exp(θ0::Real, θ1::T, θ2::Real, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real, geoid::Real, roy::Real) where {T}
    rev_exp(θ0,θ1,θ2,θ3,θ4,σ,logp,ψ,false,geoid, roy) * θ4 * _ρ(σ)
end

const STARTING_σ_ψ      = 0x1.baddbb87af68ap-2 # = 0.432
const STARTING_log_ogip = 0x1.670bf3d5b282dp-1 # = 0.701

@inline    rev_exp_restricted(θ1::T, σ::T, logp::Real, ψ::Real, Dgt0::Bool, geoid::Real, roy::Real) where {T} = rev_exp(   1, θ1, 1, STARTING_log_ogip, STARTING_σ_ψ, σ, logp, ψ, Dgt0, geoid, roy)
@inline drevdσ_exp_restricted(θ1::T, σ::T, logp::Real, ψ::Real,             geoid::Real, roy::Real) where {T} = drevdσ_exp(1, θ1, 1, STARTING_log_ogip, STARTING_σ_ψ, σ, logp, ψ,       geoid, roy)
@inline drevdψ_exp_restricted(θ1::T, σ::T, logp::Real, ψ::Real,             geoid::Real, roy::Real) where {T} = drevdψ_exp(1, θ1, 1, STARTING_log_ogip, STARTING_σ_ψ, σ, logp, ψ,       geoid, roy)

# -----------------------------------------
# Flows
# -----------------------------------------

include("more-flow-payoffs.jl")

@inline function flow(::Type{Val{:dgt1_pricebreak}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[9]
        return zero(T)
    end
    u = rev_exp(1,θ[1],θ[4+Dgt0],θ[2],θ[3],σ,first(z),ψ,Dgt0,geoid, roy) + (d==1 ? θ[6] : θ[7] ) + θ[8]*exp(z[2])
    d>1      && (u *= d)
    return u::T
end

@inline function flow(::Type{Val{:dgt1_pricebreak_restr}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[7]
        return zero(T)
    end
    u = rev_exp(1, θ[1], θ[2+Dgt0], STARTING_log_ogip, STARTING_σ_ψ, σ, first(z), ψ, Dgt0, geoid, roy) + (d==1 ? θ[4] : θ[5] ) + θ[6]*exp(z[2])
    d>1      && (u *= d)
    return u::T
end

# -----------------------------------------
# number of parms
# -----------------------------------------

function number_of_model_parms(FF::Symbol)::Int
    FF ∈ (:one_restr,)                            && return  3
    FF ∈ (:dgt1_restr,:Dgt0_restr,)               && return  4
    FF ∈ (:one,:dgt1_ext_restr, :dgt1_d1_restr,:dgt1_cost_restr)         && return  5
    FF ∈ (:dgt1,:Dgt0,:dgt1_cost_Dgt0_restr,:dgt1_pricecost_restr,)      && return  6
    FF ∈ (:dgt1_ext,:dgt1_d1,:dgt1_cost,:dgt1_pricebreak_restr,)         && return  7
    FF ∈ (:dgt1_cost_Dgt0,:dgt1_pricecost,)        && return  8
    FF ∈ (:dgt1_pricebreak,)                       && return  9
    # FF ∈ (:exproy_extend,)                      && return 10
    throw(error("FF = $(FF) not recognized"))
end

# -----------------------------------------
# dθ
# -----------------------------------------

@inline function flowdθ(::Type{Val{:dgt1_pricebreak}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T},  ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return                   d * rev_exp(1,θ[1],θ[4+Dgt0],θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return                   d * rev_exp(1,θ[1],θ[4+Dgt0],θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return                   d * rev_exp(1,θ[1],θ[4+Dgt0],θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))
    k == 4  && return  Dgt0 ? zero(T) : d * rev_exp(1,θ[1],θ[4+Dgt0],θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * first(z)
    k == 5  && return !Dgt0 ? zero(T) : d * rev_exp(1,θ[1],θ[4+Dgt0],θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * first(z)

    k == 6  && return  d  == 1 ? one(T)  : zero(T)
    k == 7  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 8  && return  d*exp(z[2])

    k == 9  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:dgt1_pricebreak_restr}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T},  ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return                   d * rev_exp(1, θ[1], θ[2+Dgt0], STARTING_log_ogip, STARTING_σ_ψ, σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return  Dgt0 ? zero(T) : d * rev_exp(1, θ[1], θ[2+Dgt0], STARTING_log_ogip, STARTING_σ_ψ, σ,z[1],ψ,Dgt0,geoid,roy) * first(z)
    k == 3  && return !Dgt0 ? zero(T) : d * rev_exp(1, θ[1], θ[2+Dgt0], STARTING_log_ogip, STARTING_σ_ψ, σ,z[1],ψ,Dgt0,geoid,roy) * first(z)

    k == 4  && return  d  == 1 ? one(T)  : zero(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 6  && return  d*exp(z[2])

    k == 7  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


# -----------------------------------------
# dσ
# -----------------------------------------



@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:dgt1_pricecost}} }, N, T}
    d == 0 && return zero(T)
    return d * drevdσ_exp(1,θ[1],θ[4],θ[2],θ[3],σ,first(z),ψ,geoid,roy)
end


@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T},  ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:dgt1_pricecost_restr}} },N, T}
    d == 0 && return zero(T)
    return d * drevdσ_exp(1, θ[1], θ[2], STARTING_log_ogip, STARTING_σ_ψ, σ, first(z), ψ, geoid, roy)
end




# can do this b/c only needed for !Dgt0
@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:dgt1_pricebreak}} }, N, T}
    d == 0 && return zero(T)
    Dgt0 = false
    return d * drevdσ_exp(1,θ[1],θ[4+Dgt0],θ[2],θ[3],σ,first(z),ψ,geoid,roy)
end


@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T},  ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:dgt1_pricebreak_restr}} },N, T}
    d == 0 && return zero(T)
    Dgt0 = false
    return d * drevdσ_exp(1, θ[1], θ[2+Dgt0], STARTING_log_ogip, STARTING_σ_ψ, σ, first(z), ψ, geoid, roy)
end


# -----------------------------------------
# dψ
# -----------------------------------------


@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:dgt1_pricecost}} }, N, T}
    d == 0  && return zero(T) # sgn_ext ? θ[10] : zero(T)
    return (d * drevdψ_exp(1,θ[1],θ[4],θ[2],θ[3],σ,first(z),ψ,geoid,roy))::T
end



@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:dgt1_pricecost_restr}} }, N, T}
    d == 0  && return zero(T) # sgn_ext ? θ[10] : zero(T)
    return (d * drevdψ_exp(1,θ[1],θ[2], STARTING_log_ogip, STARTING_σ_ψ,σ,first(z),ψ,geoid,roy))::T
end



@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:dgt1_pricebreak}} }, N, T}
    d == 0  && return zero(T) # sgn_ext ? θ[10] : zero(T)
    Dgt0 = false
    return (d * drevdψ_exp(1,θ[1],θ[4+Dgt0],θ[2],θ[3],σ,first(z),ψ,geoid,roy))::T
end



@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:dgt1_pricebreak_restr}} }, N, T}
    d == 0  && return zero(T) # sgn_ext ? θ[10] : zero(T)
    Dgt0 = false
    return (d * drevdψ_exp(1,θ[1],θ[2+Dgt0], STARTING_log_ogip, STARTING_σ_ψ,σ,first(z),ψ,geoid,roy))::T
end
