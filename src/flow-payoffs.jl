export flow, flowdθ, flowdσ, flowdψ, STARTING_σ_ψ, STARTING_log_ogip

const STARTING_σ_ψ      = 0x1.baddbb87af68ap-2 # = 0.432
const STARTING_log_ogip = 0x1.670bf3d5b282dp-1 # = 0.701
const STARTING_t = 0.042/(2016-2003)

# functions in case we have months to expiration
@inline flow(  FF::Type, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, τ::Real, geoid::Real, roy::T) where {N,T} = flow(  FF, θ, σ, z, ψ,    d, d1, Dgt0, sgn_ext, geoid, roy)
@inline flowdθ(FF::Type, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, k::Integer, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, τ::Real, geoid::Real, roy::T) where {N,T} = flowdθ(FF, θ, σ, z, ψ, k, d, d1, Dgt0, sgn_ext, geoid, roy)
@inline flowdσ(FF::Type, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, τ::Real, geoid::Real, roy::T) where {N,T} = flowdσ(FF, θ, σ, z, ψ,    d,                    geoid, roy)
@inline flowdσ(FF::Type, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T,             d::Integer,                                                     geoid::Real, roy::T) where {N,T} = flowdσ(FF, θ, σ, z, ψ,    d,                    geoid, roy)

@inline flowdψ(FF::Type, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T,             d::Integer,                          sgn_ext::Bool, τ::Real, geoid::Real, roy::T) where {N,T} = flowdψ(FF, θ, σ, z, ψ,    d,           sgn_ext, geoid, roy)
@inline flowdψ(FF::Type, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, τ::Real, geoid::Real, roy::T) where {N,T} = flowdψ(FF, θ, σ, z, ψ,    d,           sgn_ext, geoid, roy)

# --------------------------- common revenue functions & derivatives  --------------------------------------

@inline    rev_exp_restricted(θ1::T, σ::T, logp::Real,          ψ::Real, Dgt0::Bool, geoid::Real, roy::Real) where {T} = rev_exp(   1, θ1, 1, STARTING_log_ogip, STARTING_σ_ψ,             σ, logp,    ψ, Dgt0, geoid, roy)
@inline drevdσ_exp_restricted(θ1::T, σ::T, logp::Real,          ψ::Real,             geoid::Real, roy::Real) where {T} = drevdσ_exp(1, θ1, 1, STARTING_log_ogip, STARTING_σ_ψ,             σ, logp,    ψ,       geoid, roy)
@inline drevdψ_exp_restricted(θ1::T, σ::T, logp::Real,          ψ::Real,             geoid::Real, roy::Real) where {T} = drevdψ_exp(1, θ1, 1, STARTING_log_ogip, STARTING_σ_ψ,             σ, logp,    ψ,       geoid, roy)

@inline    rev_exp_restricted(θ1::T, σ::T, logp::Real, t::Real, ψ::Real, Dgt0::Bool, geoid::Real, roy::Real) where {T} = rev_exp(   1, θ1, 1, STARTING_log_ogip, STARTING_σ_ψ, STARTING_t, σ, logp, t, ψ, Dgt0, geoid, roy)
@inline drevdσ_exp_restricted(θ1::T, σ::T, logp::Real, t::Real, ψ::Real,             geoid::Real, roy::Real) where {T} = drevdσ_exp(1, θ1, 1, STARTING_log_ogip, STARTING_σ_ψ, STARTING_t, σ, logp, t, ψ,       geoid, roy)
@inline drevdψ_exp_restricted(θ1::T, σ::T, logp::Real, t::Real, ψ::Real,             geoid::Real, roy::Real) where {T} = drevdψ_exp(1, θ1, 1, STARTING_log_ogip, STARTING_σ_ψ, STARTING_t, σ, logp, t, ψ,       geoid, roy)

# chebshev polynomials
# See http://www.aip.de/groups/soe/local/numres/bookcpdf/c5-8.pdf
@inline checkinterval(x::Real,min::Real,max::Real) =  min <= x <= max || throw(DomainError("x = $x must be in [$min,$max]"))
@inline checkinterval(x::Real) = checkinterval(x,-1,1)
@inline cheb0(x::Real) = (checkinterval(x); return one(Float64))
@inline cheb1(x::Real) = (checkinterval(x); return x)
@inline cheb2(x::Real) = (checkinterval(x); return 2*x^2 - 1)
@inline cheb3(x::Real) = (checkinterval(x); return 4*x^3 - 3*x)
@inline cheb4(x::Real) = (checkinterval(x); return 8*(x^4 - x^2) + 1)

# -----------------------------------------
# Flows
# -----------------------------------------

include("flow-payoffs-more.jl")

@inline function flow(::Type{Val{:cheb3_cost_tech}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[11]
        return zero(T)
    end
    logp, logc, t = z
    u = rev_exp(1,θ[1],1,θ[2],θ[3],θ[4],σ,logp,t,ψ,Dgt0,geoid, roy) + (d==1 ? θ[5] : θ[6] )*cheb0(t) + θ[7]*cheb1(t) + θ[8]*cheb2(t) + θ[9]*cheb3(t) + θ[10]*exp(logc)
    d>1 && (u *= d)
    return u::T
end

@inline function flow(::Type{Val{:cheb3_cost_tech_restr}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[8]
        return zero(T)
    end
    logp, logc, t = z
    u = rev_exp_restricted(θ[1], σ, logp, t, ψ, Dgt0, geoid, roy) + (d==1 ? θ[2] : θ[3] )*cheb0(t) + θ[4]*cheb1(t) + θ[5]*cheb2(t) + θ[6]*cheb3(t) + θ[7]*exp(logc)
    d>1 && (u *= d)
    return u::T
end


# -----------------------------------------
# number of parms
# -----------------------------------------

function number_of_model_parms(FF::Symbol)::Int
    FF ∈ (:one_restr,)                                                           && return  3
    FF ∈ (:dgt1_restr,:Dgt0_restr,)                                              && return  4
    FF ∈ (:one,:dgt1_ext_restr, :dgt1_d1_restr,:dgt1_cost_restr,:cheb2_restr)   && return  5
    FF ∈ (:dgt1,:Dgt0,:dgt1_cost_Dgt0_restr,:dgt1_pricecost_restr,:cheb3_restr) && return  6
    FF ∈ (:dgt1_ext,:dgt1_d1,:dgt1_cost,:dgt1_pricebreak_restr,:cheb2, :cheb3_dgt1_restr,)  && return  7
    FF ∈ (:dgt1_cost_Dgt0,:dgt1_pricecost,:cheb3,:cheb3_cost_restr,:ttbuild_cost_restr,:cheb3_cost_tech_restr,)  && return  8
    FF ∈ (:dgt1_pricebreak,:cheb3_dgt1,)                                         && return  9
    FF ∈ (:cheb3_cost,:ttbuild_cost,)                                            && return 10
    FF ∈ (:cheb3_cost_tech,)                                                     && return 11
    throw(error("FF = $(FF) not recognized"))
end

# -----------------------------------------
# dθ
# -----------------------------------------

@inline function flowdθ(::Type{Val{:cheb3_cost_tech}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    logp, logc, t = z

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],θ[4],σ,logp,t,ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],θ[4],σ,logp,t,ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],θ[4],σ,logp,t,ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))
    k == 4  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],θ[4],σ,logp,t,ψ,Dgt0,geoid,roy) * t

    # drilling cost
    k == 5  && return d != 1 ? zero(T) : T(   cheb0(t) )
    k == 6  && return d <= 1 ? zero(T) : T( d*cheb0(t) )
    k == 7  && return d == 0 ? zero(T) : T( d*cheb1(t) )
    k == 8  && return d == 0 ? zero(T) : T( d*cheb2(t) )
    k == 9  && return d == 0 ? zero(T) : T( d*cheb3(t) )
    k == 10  && return d == 0 ? zero(T) : T( d*exp(logc) )

    # extension cost
    k == 11  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:cheb3_cost_tech_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    logp, logc, t = z

    # revenue
    k == 1  && return   d * rev_exp_restricted(θ[1], σ, logp, t, ψ, Dgt0, geoid, roy)

    # drilling cost
    k == 2  && return d != 1 ? zero(T) : T(   cheb0(t) )
    k == 3  && return d <= 1 ? zero(T) : T( d*cheb0(t) )
    k == 4  && return d == 0 ? zero(T) : T( d*cheb1(t) )
    k == 5  && return d == 0 ? zero(T) : T( d*cheb2(t) )
    k == 6  && return d == 0 ? zero(T) : T( d*cheb3(t) )
    k == 7  && return d == 0 ? zero(T) : T( d*exp(logc) )
    # extension cost
    k == 8  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end

# -----------------------------------------
# dσ
# -----------------------------------------


@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:one}}, Type{Val{:dgt1}}, Type{Val{:Dgt0}}, Type{Val{:dgt1_ext}}, Type{Val{:dgt1_d1}}, Type{Val{:dgt1_cost}}, Type{Val{:dgt1_cost_Dgt0}}, Type{Val{:cheb2}}, Type{Val{:cheb3}}, Type{Val{:cheb3_dgt1}}, Type{Val{:cheb3_cost}}, Type{Val{:ttbuild_cost}} }, N, T}
    d == 0 && return zero(T)
    return d * drevdσ_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,geoid,roy)
end


@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T},  ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:one_restr}}, Type{Val{:dgt1_restr}}, Type{Val{:Dgt0_restr}}, Type{Val{:dgt1_ext_restr}}, Type{Val{:dgt1_d1_restr}}, Type{Val{:dgt1_cost_restr}}, Type{Val{:dgt1_cost_Dgt0_restr}}, Type{Val{:cheb2_restr}}, Type{Val{:cheb3_restr}}, Type{Val{:cheb3_dgt1_restr}}, Type{Val{:cheb3_cost_restr}}, Type{Val{:ttbuild_cost_restr}} },N, T}
    d == 0 && return zero(T)
    return d * drevdσ_exp_restricted(θ[1],σ,z[1],ψ,geoid,roy)
end



@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:cheb3_cost_tech}} }, N, T}
    d == 0 && return zero(T)
    return d * drevdσ_exp(1,θ[1],1,θ[2],θ[3],θ[4],σ,first(z),last(z),ψ,geoid,roy)
end


@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T},  ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:cheb3_cost_tech_restr}}  },N, T}
    d == 0 && return zero(T)
    return d * drevdσ_exp_restricted(θ[1],σ,first(z),last(z),ψ,geoid,roy)
end


# -----------------------------------------
# dψ
# -----------------------------------------

@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:one}}, Type{Val{:dgt1}}, Type{Val{:Dgt0}}, Type{Val{:dgt1_d1}}, Type{Val{:dgt1_cost}}, Type{Val{:dgt1_cost_Dgt0}}, Type{Val{:cheb2}}, Type{Val{:cheb3}}, Type{Val{:cheb3_dgt1}}, Type{Val{:cheb3_cost}}, Type{Val{:ttbuild_cost}} },N, T}
    d == 0  && return zero(T) # sgn_ext ? θ[10] : zero(T)
    return (d * drevdψ_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,geoid,roy))::T
end


@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:one_restr}}, Type{Val{:dgt1_restr}}, Type{Val{:Dgt0_restr}}, Type{Val{:dgt1_d1_restr}}, Type{Val{:dgt1_cost_restr}}, Type{Val{:dgt1_cost_Dgt0_restr}}, Type{Val{:cheb2_restr}}, Type{Val{:cheb3_restr}}, Type{Val{:cheb3_dgt1_restr}}, Type{Val{:cheb3_cost_restr}}, Type{Val{:ttbuild_cost_restr}} },N, T}
    d == 0  && return zero(T) # sgn_ext ? θ[10] : zero(T)
    return (d * drevdψ_exp_restricted(θ[1],σ,z[1],ψ,geoid,roy))::T
end


@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:cheb3_cost_tech}} }, N, T}
    d == 0  && return zero(T) # sgn_ext ? θ[10] : zero(T)
    return (d * drevdψ_exp(1,θ[1],1,θ[2],θ[3],θ[4],σ,first(z),last(z),ψ,geoid,roy))::T
end


@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:cheb3_cost_tech}} }, N, T}
    d == 0  && return zero(T) # sgn_ext ? θ[10] : zero(T)
    return (d * drevdψ_exp_restricted(θ[1],σ,first(z),last(z),ψ,geoid,roy))::T
end
