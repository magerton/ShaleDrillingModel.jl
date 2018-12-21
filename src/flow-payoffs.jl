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

# Chebyshev polynomials
# See http://www.aip.de/groups/soe/local/numres/bookcpdf/c5-8.pdf
@inline checkinterval(x::Real,min::Real,max::Real) =  min <= x <= max || throw(DomainError("x = $x must be in [$min,$max]"))
@inline checkinterval(x::Real) = checkinterval(x,-1,1)
@inline cheby0(x::Real) = (checkinterval(x); return one(Float64))
@inline cheby1(x::Real) = (checkinterval(x); return x)
@inline cheby2(x::Real) = (checkinterval(x); return 2*x^2 - 1)
@inline cheby3(x::Real) = (checkinterval(x); return 4*x^3 - 3*x)
@inline cheby4(x::Real) = (checkinterval(x); return 8*(x^4 - x^2) + 1)

# cheby03(x) = (cheby0(x), cheby1(x), cheby2(x), cheby3(x))
# cheby03(0.2)


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


@inline function flow(::Type{Val{:cheby2}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[7]
        return zero(T)
    end
    x = last(z)
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,first(z),ψ,Dgt0,geoid, roy) + θ[4]*cheby0(x) + θ[5]*cheby1(x) + θ[6]*cheby2(x)
    d>1 && (u *= d)
    return u::T
end

@inline function flow(::Type{Val{:cheby2_restr}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[5]
        return zero(T)
    end
    x = last(z)
    u = rev_exp_restricted(θ[1], σ, first(z), ψ, Dgt0, geoid, roy) + θ[2]*cheby0(x) + θ[3]*cheby1(x) + θ[4]*cheby2(x)
    d>1 && (u *= d)
    return u::T
end



@inline function flow(::Type{Val{:cheby3}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[8]
        return zero(T)
    end
    x = last(z)
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,first(z),ψ,Dgt0,geoid, roy) + θ[4]*cheby0(x) + θ[5]*cheby1(x) + θ[6]*cheby2(x) + θ[7]*cheby3(x)
    d>1 && (u *= d)
    return u::T
end

@inline function flow(::Type{Val{:cheby3_restr}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[6]
        return zero(T)
    end
    x = last(z)
    u = rev_exp_restricted(θ[1], σ, first(z), ψ, Dgt0, geoid, roy) + θ[2]*cheby0(x) + θ[3]*cheby1(x) + θ[4]*cheby2(x) + θ[5]*cheby3(x)
    d>1 && (u *= d)
    return u::T
end



@inline function flow(::Type{Val{:cheby3_dgt1}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[9]
        return zero(T)
    end
    x = last(z)
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,first(z),ψ,Dgt0,geoid, roy) + (d==1 ? θ[4] : θ[5] )*cheby0(x) + θ[6]*cheby1(x) + θ[7]*cheby2(x) + θ[8]*cheby3(x)
    d>1 && (u *= d)
    return u::T
end

@inline function flow(::Type{Val{:cheby3_dgt1_restr}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[7]
        return zero(T)
    end
    x = last(z)
    u = rev_exp_restricted(θ[1], σ, first(z), ψ, Dgt0, geoid, roy) + (d==1 ? θ[2] : θ[3] )*cheby0(x) + θ[4]*cheby1(x) + θ[5]*cheby2(x) + θ[6]*cheby3(x)
    d>1 && (u *= d)
    return u::T
end




# -----------------------------------------
# number of parms
# -----------------------------------------

function number_of_model_parms(FF::Symbol)::Int
    FF ∈ (:one_restr,)                                                           && return  3
    FF ∈ (:dgt1_restr,:Dgt0_restr,)                                              && return  4
    FF ∈ (:one,:dgt1_ext_restr, :dgt1_d1_restr,:dgt1_cost_restr,:cheby2_restr)   && return  5
    FF ∈ (:dgt1,:Dgt0,:dgt1_cost_Dgt0_restr,:dgt1_pricecost_restr,:cheby3_restr) && return  6
    FF ∈ (:dgt1_ext,:dgt1_d1,:dgt1_cost,:dgt1_pricebreak_restr,:cheby2, :cheby3_dgt1_restr,)  && return  7
    FF ∈ (:dgt1_cost_Dgt0,:dgt1_pricecost,:cheby3,)                               && return  8
    FF ∈ (:dgt1_pricebreak,:cheby3_dgt1,)                                         && return  9
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












@inline function flowdθ(::Type{Val{:cheby2}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 4  && return d == 0 ? zero(T) : T( d*cheby0(z[2]) )
    k == 5  && return d == 0 ? zero(T) : T( d*cheby1(z[2]) )
    k == 6  && return d == 0 ? zero(T) : T( d*cheby2(z[2]) )

    # extension cost
    k == 7  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:cheby2_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    # drilling cost
    k == 2  && return d == 0 ? zero(T) : T( d*cheby0(z[2]) )
    k == 3  && return d == 0 ? zero(T) : T( d*cheby1(z[2]) )
    k == 4  && return d == 0 ? zero(T) : T( d*cheby2(z[2]) )

    # extension cost
    k == 5  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end







@inline function flowdθ(::Type{Val{:cheby3}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 4  && return d == 0 ? zero(T) : T( d*cheby0(z[2]) )
    k == 5  && return d == 0 ? zero(T) : T( d*cheby1(z[2]) )
    k == 6  && return d == 0 ? zero(T) : T( d*cheby2(z[2]) )
    k == 7  && return d == 0 ? zero(T) : T( d*cheby3(z[2]) )

    # extension cost
    k == 8  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:cheby3_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    # drilling cost
    k == 2  && return d == 0 ? zero(T) : T( d*cheby0(z[2]) )
    k == 3  && return d == 0 ? zero(T) : T( d*cheby1(z[2]) )
    k == 4  && return d == 0 ? zero(T) : T( d*cheby2(z[2]) )
    k == 5  && return d == 0 ? zero(T) : T( d*cheby3(z[2]) )

    # extension cost
    k == 6  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end





@inline function flowdθ(::Type{Val{:cheby3_dgt1}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 4  && return d == 1 ? T( cheby0(z[2]) ) : zero(T)
    k == 5  && return d <= 1 ? zero(T) : T( d*cheby0(z[2]) )
    k == 6  && return d == 0 ? zero(T) : T( d*cheby1(z[2]) )
    k == 7  && return d == 0 ? zero(T) : T( d*cheby2(z[2]) )
    k == 8  && return d == 0 ? zero(T) : T( d*cheby3(z[2]) )

    # extension cost
    k == 9  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:cheby3_dgt1_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    # drilling cost
    k == 2  && return d == 1 ? T( cheby0(z[2]) ) : zero(T)
    k == 3  && return d <= 1 ? zero(T) : T( d*cheby0(z[2]) )
    k == 4  && return d == 0 ? zero(T) : T( d*cheby1(z[2]) )
    k == 5  && return d == 0 ? zero(T) : T( d*cheby2(z[2]) )
    k == 6  && return d == 0 ? zero(T) : T( d*cheby3(z[2]) )

    # extension cost
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
