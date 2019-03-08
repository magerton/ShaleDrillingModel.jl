export flow, flowdθ, flowdσ, flowdψ, STARTING_σ_ψ, STARTING_log_ogip, STARTING_t

const STARTING_σ_ψ      = 0.349148 # 0x1.baddbb87af68ap-2 # = 0.432
const STARTING_log_ogip = 0.709267 # 0x1.670bf3d5b282dp-1 # = 0.701
const STARTING_t = 2*0.042/(2016-2003)

# functions in case we have months to expiration
@inline flow(  FF::Type, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, τ::Real, geoid::Real, roy::T) where {N,T} = flow(  FF, θ, σ, z, ψ,    d, d1, Dgt0, sgn_ext, geoid, roy)
@inline flowdθ(FF::Type, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T, k::Integer, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, τ::Real, geoid::Real, roy::T) where {N,T} = flowdθ(FF, θ, σ, z, ψ, k, d, d1, Dgt0, sgn_ext, geoid, roy)
@inline flowdσ(FF::Type, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, τ::Real, geoid::Real, roy::T) where {N,T} = flowdσ(FF, θ, σ, z, ψ,    d,                    geoid, roy)
@inline flowdψ(FF::Type, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T,             d::Integer,                          sgn_ext::Bool, τ::Real, geoid::Real, roy::T) where {N,T} = flowdψ(FF, θ, σ, z, ψ,    d,           sgn_ext, geoid, roy)
@inline flowdψ(FF::Type, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, τ::Real, geoid::Real, roy::T) where {N,T} = flowdψ(FF, θ, σ, z, ψ,    d,           sgn_ext, geoid, roy)
@inline flowdσ(FF::Type, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T,             d::Integer,                                                  geoid::Real, roy::T) where {N,T} = flowdσ(FF, θ, σ, z, ψ,    d,                    geoid, roy)


# functions in case we have months to expiration
@inline flow(  FF::Type, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T,             d::Integer, geoid::Real, roy::T) where {N,T} = flow(  FF, θ, σ, z, ψ,    d, _d1(wp,i), _Dgt0(wp,i), _sgnext(wp,i), geoid, roy)
@inline flowdθ(FF::Type, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T, k::Integer, d::Integer, geoid::Real, roy::T) where {N,T} = flowdθ(FF, θ, σ, z, ψ, k, d, _d1(wp,i), _Dgt0(wp,i), _sgnext(wp,i), geoid, roy)
@inline flowdσ(FF::Type, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T,             d::Integer, geoid::Real, roy::T) where {N,T} = flowdσ(FF, θ, σ, z, ψ,    d,                                        geoid, roy)
@inline flowdψ(FF::Type, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T,             d::Integer, geoid::Real, roy::T) where {N,T} = flowdψ(FF, θ, σ, z, ψ,    d,                         _sgnext(wp,i), geoid, roy)

# @inline function flowdθ(::Type, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, k::Integer, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}

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

@inline cheb0(z::Real) = (x = clamp(z,-1,1); return one(eltype(z)))
@inline cheb1(z::Real) = (x = clamp(z,-1,1); return x)
@inline cheb2(z::Real) = (x = clamp(z,-1,1); return 2*x^2 - 1)
@inline cheb3(z::Real) = (x = clamp(z,-1,1); return 4*x^3 - 3*x)
@inline cheb4(z::Real) = (x = clamp(z,-1,1); return 8*(x^4 - x^2) + 1)

# -----------------------------------------
# Flows
# -----------------------------------------

include("flow-payoffs-more.jl")

#      (2008,2009,2010,2011,2012)
#  4 + (0  , 1  , 2  , 3  , 4)
@inline θk(   model::FF, t::Integer) where {FF<:Union{ Type{Val{:timefe0812}}, Type{Val{:timefecost0812}}, Type{Val{:timefe0812_restr}}, Type{Val{:timefecost0812_restr}} } } = clamp(t,2008,2012)-2008
@inline θkmax(model::FF)             where {FF<:Union{ Type{Val{:timefe0812}}, Type{Val{:timefecost0812}}, Type{Val{:timefe0812_restr}}, Type{Val{:timefecost0812_restr}} } } = 4


@inline function flow(model::FF, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple{Float64,<:Integer}, ψ::T, d::Integer, geoid::Real, roy::T)::T where {T,FF<:Union{Type{Val{:timefe0812}}}}
    if d == 0
        _sgnext(wp,i) && return θ[4+θkmax(model)+2]
        return zero(T)
    end
    logp, t = z
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,_Dgt0(wp,i),geoid,roy) + θ[4+θk(model,t)]
    if d>1
        u += θ[4+θkmax(model)+1]
        u *= d
    end
    return u::T
end


@inline function flow(model::FF, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple{Float64,Float64,<:Integer}, ψ::T, d::Integer, geoid::Real, roy::T)::T where {T,FF<:Union{Type{Val{:timefecost0812}}}}
    if d == 0
        _sgnext(wp,i) && return θ[4+θkmax(model)+3]
        return zero(T)
    end
    logp, logc, t = z
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,_Dgt0(wp,i),geoid,roy) + θ[4+θk(model,t)] + θ[4+θkmax(model)+2]*exp(logc)
    if d>1
        u += θ[4+θkmax(model)+1]
        u *= d
    end
    return u::T
end


@inline function flow(model::FF, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple{Float64,<:Integer}, ψ::T, d::Integer, geoid::Real, roy::T)::T where {T,FF<:Union{Type{Val{:timefe0812_restr}}}}
        if d == 0
            _sgnext(wp,i) && return θ[2+θkmax(model)+2]
            return zero(T)
        end
        logp, t = z
        u = rev_exp_restricted(θ[1], σ, logp, ψ, _Dgt0(wp,i), geoid, roy) + θ[2+θk(model,t)]
        if d>1
            u += θ[2+θkmax(model)+1]
            u *= d
        end
        return u::T
    end


@inline function flow(model::FF, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple{Float64,Float64,<:Integer}, ψ::T, d::Integer, geoid::Real, roy::T)::T where {T,FF<:Union{Type{Val{:timefecost0812_restr}}}}
    if d == 0
        _sgnext(wp,i) && return θ[2+θkmax(model)+3]
        return zero(T)
    end
    logp, logc, t = z
    u = rev_exp_restricted(θ[1], σ, logp, ψ, _Dgt0(wp,i), geoid, roy) + θ[2+θk(model,t)] + θ[2+θkmax(model)+2]*exp(logc)
    if d>1
        u += θ[2+θkmax(model)+1]
        u *= d
    end
    return u::T
end
# -----------------------------------------
# number of parms
# -----------------------------------------

function number_of_model_parms(FF::Symbol)::Int
    FF ∈ (:one_restr,)                                                          && return  3
    FF ∈ (:dgt1_restr,:Dgt0_restr,)                                             && return  4
    FF ∈ (:one,:dgt1_ext_restr, :dgt1_d1_restr,:dgt1_cost_restr,:cheb2_restr)   && return  5
    FF ∈ (:dgt1,:Dgt0,:dgt1_cost_Dgt0_restr,:dgt1_pricecost_restr,:cheb3_restr) && return  6
    FF ∈ (:dgt1_ext,:dgt1_d1,:dgt1_cost,:dgt1_pricebreak_restr,:cheb2, :cheb3_dgt1_restr,)                                         && return  7
    FF ∈ (:dgt1_cost_Dgt0,:dgt1_pricecost,:cheb3,:cheb3_cost_restr,:ttbuild_cost_restr,:cheb3_cost_tech_restr,:timefe0812_restr,)  && return  8
    FF ∈ (:dgt1_pricebreak,:cheb3_dgt1,:timefecost0812_restr,)                  && return  9
    FF ∈ (:cheb3_cost,:ttbuild_cost,:timefe0812,)                              && return 10
    FF ∈ (:cheb3_cost_tech,:timefecost0812,)                                    && return 11

    throw(error("FF = $(FF) not recognized"))
end

# -----------------------------------------
# dθ
# -----------------------------------------

@inline function flowdθ(model::FF, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple{Float64,<:Integer}, ψ::T, k::Integer, d::Integer, geoid::Real, roy::T)::T where {T,FF<:Union{Type{Val{:timefe0812}}}}
    d == 0 && !_sgnext(wp,i) && return zero(T)
    Dgt0 = _Dgt0(wp,i)
    logp, t = z

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    if k <= 4+θkmax(model)
        k == 4+θk(model,t) && return d == 0 ? zero(T) : T(d)
        return zero(T)
    end
    k == 4+θkmax(model)+1 && return d <= 1 ? zero(T) : T(d)

    # extension cost
    k == 4+θkmax(model)+2  && return d == 0 && _sgnext(wp,i) ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(model::FF, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple{Float64,Float64,<:Integer}, ψ::T, k::Integer, d::Integer, geoid::Real, roy::T)::T where {T,FF<:Union{Type{Val{:timefecost0812}}}}
    d == 0 && !_sgnext(wp,i) && return zero(T)
    Dgt0 = _Dgt0(wp,i)
    logp, logc, t = z

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    if k <= 4+θkmax(model)
        k == 4+θk(model,t)    && return d == 0 ? zero(T) : T(d)
        return zero(T)
    end
    k == 4+θkmax(model)+1 && return d <= 1 ? zero(T) : T(d)
    k == 4+θkmax(model)+2 && return d == 0 ? zero(T) : T(d*exp(logc))

    # extension cost
    k == 4+θkmax(model)+3  && return d == 0 && _sgnext(wp,i) ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(model::FF, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple{Float64,<:Integer}, ψ::T, k::Integer, d::Integer, geoid::Real, roy::T)::T where {T,FF<:Union{Type{Val{:timefe0812_restr}}}}
    d == 0 && !_sgnext(wp,i) && return zero(T)
    Dgt0 = _Dgt0(wp,i)
    logp, t = z

    # revenue
    k == 1  && return   d * rev_exp_restricted(θ[1], σ, logp, ψ, _Dgt0(wp,i), geoid, roy)

    # drilling cost
    if k <= 2+θkmax(model)
        k == 2+θk(model,t)    && return d == 0 ? zero(T) : T(d)
        return zero(T)
    end
    k == 2+θkmax(model)+1 && return d <= 1 ? zero(T) : T(d)

    # extension cost
    k == 2+θkmax(model)+2  && return d == 0 && _sgnext(wp,i) ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(model::FF, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple{Float64,Float64,<:Integer}, ψ::T, k::Integer, d::Integer, geoid::Real, roy::T)::T where {T,FF<:Union{Type{Val{:timefecost0812_restr}}}}
    d == 0 && !_sgnext(wp,i) && return zero(T)
    Dgt0 = _Dgt0(wp,i)
    logp, logc, t = z

    # revenue
    k == 1  && return   d * rev_exp_restricted(θ[1], σ, logp, ψ, _Dgt0(wp,i), geoid, roy)

    # drilling cost
    if k <= 2+θkmax(model)
        k == 2+θk(model,t)    && return d == 0 ? zero(T) : T(d)
        return zero(T)
    end
    k == 2+θkmax(model)+1 && return d <= 1 ? zero(T) : T(d)
    k == 2+θkmax(model)+2 && return d == 0 ? zero(T) : T(d*exp(logc))

    # extension cost
    k == 2+θkmax(model)+3  && return d == 0 && _sgnext(wp,i) ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end

# -----------------------------------------
# dσ
# -----------------------------------------


@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:one}}, Type{Val{:dgt1}}, Type{Val{:Dgt0}}, Type{Val{:dgt1_ext}}, Type{Val{:dgt1_d1}}, Type{Val{:dgt1_cost}}, Type{Val{:dgt1_cost_Dgt0}}, Type{Val{:cheb2}}, Type{Val{:cheb3}}, Type{Val{:cheb3_dgt1}}, Type{Val{:cheb3_cost}}, Type{Val{:ttbuild_cost}}, Type{Val{:timefecost0812}}, Type{Val{:timefe0812}} }, T}
    d == 0 && return zero(T)
    return d * drevdσ_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,geoid,roy)
end


@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, z::Tuple,  ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:one_restr}}, Type{Val{:dgt1_restr}}, Type{Val{:Dgt0_restr}}, Type{Val{:dgt1_ext_restr}}, Type{Val{:dgt1_d1_restr}}, Type{Val{:dgt1_cost_restr}}, Type{Val{:dgt1_cost_Dgt0_restr}}, Type{Val{:cheb2_restr}}, Type{Val{:cheb3_restr}}, Type{Val{:cheb3_dgt1_restr}}, Type{Val{:cheb3_cost_restr}}, Type{Val{:ttbuild_cost_restr}}, Type{Val{:timefecost0812_restr}}, Type{Val{:timefe0812_restr}} }, T}
    d == 0 && return zero(T)
    return d * drevdσ_exp_restricted(θ[1],σ,z[1],ψ,geoid,roy)
end


@inline function flowdσ(::FF, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:cheb3_cost_tech}} }, T}
    d == 0 && return zero(T)
    return d * drevdσ_exp(1,θ[1],1,θ[2],θ[3],θ[4],σ,first(z),last(z),ψ,geoid,roy)
end


@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, z::Tuple,  ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:cheb3_cost_tech_restr}}  }, T}
    d == 0 && return zero(T)
    return d * drevdσ_exp_restricted(θ[1],σ,first(z),last(z),ψ,geoid,roy)
end


# -----------------------------------------
# dψ
# -----------------------------------------

@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:one}}, Type{Val{:dgt1}}, Type{Val{:Dgt0}}, Type{Val{:dgt1_d1}}, Type{Val{:dgt1_cost}}, Type{Val{:dgt1_cost_Dgt0}}, Type{Val{:cheb2}}, Type{Val{:cheb3}}, Type{Val{:cheb3_dgt1}}, Type{Val{:cheb3_cost}}, Type{Val{:ttbuild_cost}}, Type{Val{:timefecost0812}}, Type{Val{:timefe0812}} }, T}
    d == 0  && return zero(T) # sgn_ext ? θ[10] : zero(T)
    return (d * drevdψ_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,geoid,roy))::T
end


@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:one_restr}}, Type{Val{:dgt1_restr}}, Type{Val{:Dgt0_restr}}, Type{Val{:dgt1_d1_restr}}, Type{Val{:dgt1_cost_restr}}, Type{Val{:dgt1_cost_Dgt0_restr}}, Type{Val{:cheb2_restr}}, Type{Val{:cheb3_restr}}, Type{Val{:cheb3_dgt1_restr}}, Type{Val{:cheb3_cost_restr}}, Type{Val{:ttbuild_cost_restr}}, Type{Val{:timefecost0812_restr}}, Type{Val{:timefe0812_restr}} }, T}
    d == 0  && return zero(T) # sgn_ext ? θ[10] : zero(T)
    return (d * drevdψ_exp_restricted(θ[1],σ,z[1],ψ,geoid,roy))::T
end

# @inline function flow(  FF::Type, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T,             d::Integer, geoid::Real, roy::T)
# @inline function flowdθ(FF::Type, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, k::Integer, d::Integer, geoid::Real, roy::T)
# @inline function flowdσ(FF::Type, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T,             d::Integer, geoid::Real, roy::T)
# @inline function flowdψ(FF::Type, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T,             d::Integer, geoid::Real, roy::T)


@inline function flowdψ(::FF, wp::AbstractUnitProblem, i::Integer, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T, d::Integer, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:cheb3_cost_tech}} },T}
    d == 0  && return zero(T) # _sgnext(wp,i) ? θ[10] : zero(T)
    return (d * drevdψ_exp(1,θ[1],1,θ[2],θ[3],θ[4],σ,first(z),last(z),ψ,geoid,roy))::T
end


@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::Tuple, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:cheb3_cost_tech_restr}} }, T}
    d == 0  && return zero(T) # sgn_ext ? θ[10] : zero(T)
    return (d * drevdψ_exp_restricted(θ[1],σ,first(z),last(z),ψ,geoid,roy))::T
end
