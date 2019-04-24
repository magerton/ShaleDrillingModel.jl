export flow, flowdθ, flowdσ, flowdψ,
    STARTING_α_ψ, STARTING_log_ogip, STARTING_t, set_STARTING_α_ψ, set_STARTING_log_ogip, set_STARTING_t,
    AbstractPayoffFunction,
    AbstractPayoffComponent,
    AbstractDrillingCost,
    AbstractDrillingCost_TimeFE,
    DrillingCost_TimeFE,
    DrillingCost_TimeFE_rigrate,
    DrillingCost_constant,
    DrillingCost_dgt1,
    AbstractExtensionCost,
    ExtensionCost_Constant,
    ExtensionCost_Zero,
    ExtensionCost_ψ,
    AbstractStaticPayoffs,
    StaticDrillingPayoff,
    ConstrainedProblem,
    UnconstrainedProblem,
    AbstractModelVariations,
    AbstractTaxType,
    NoTaxes,
    WithTaxes,
    AbstractTechChange,
    NoTrend,
    TimeTrend,
    AbstractConstrainedType,
    Unconstrained,
    Constrained,
    DrillingRevenue

using InteractiveUtils: subtypes

import Base: length

# -----------------------------------------
# some constants
# -----------------------------------------

# From Gulen et al (2015) "Production scenarios for the Haynesville..."
const GATH_COMP_TRTMT_PER_MCF   = 0.42 + 0.07
const MARGINAL_TAX_RATE = 0.42
const ONE_MINUS_MARGINAL_TAX_RATE = 1 - MARGINAL_TAX_RATE

# other calculations
const REAL_DISCOUNT_AND_DECLINE = 0x1.89279c9f3217dp-1   # computed from time FE in monthly pdxn
const C_PER_MCF = GATH_COMP_TRTMT_PER_MCF * REAL_DISCOUNT_AND_DECLINE

const STARTING_α_ψ      = 0x1.7587cc6793516p-2 # 0.365
const STARTING_log_ogip = 0x1.401755c339009p-1 # 0.625
const STARTING_t        = 0.02

# -----------------------------------------
# some functions
# -----------------------------------------

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
# big types
# -----------------------------------------

function showtypetree(T, level=0)
    println("\t" ^ level, T)
    for t in subtypes(T)
        showtypetree(t, level+1)
   end
end

# Static Payoff
abstract type AbstractPayoffFunction end
abstract type AbstractStaticPayoffs   <: AbstractPayoffFunction end
abstract type AbstractPayoffComponent <: AbstractPayoffFunction end

# payoff components
abstract type AbstractDrillingRevenue <: AbstractPayoffComponent end
abstract type AbstractDrillingCost    <: AbstractPayoffComponent end
abstract type AbstractExtensionCost   <: AbstractPayoffComponent end

# -------------------------------------------
# Drilling payoff has 3 parts
# -------------------------------------------

struct StaticDrillingPayoff{R<:AbstractDrillingRevenue,C<:AbstractDrillingCost,E<:AbstractExtensionCost} <: AbstractStaticPayoffs
    revenue::R
    drillingcost::C
    extensioncost::E
end

revenue(x::StaticDrillingPayoff) = x.revenue
drillingcost(x::StaticDrillingPayoff) = x.drillingcost
extensioncost(x::StaticDrillingPayoff) = x.extensioncost

@inline length(x::StaticDrillingPayoff) = length(x.revenue) + length(x.drillingcost) + length(x.extensioncost)
@inline lengths(x::StaticDrillingPayoff) = (length(x.revenue), length(x.drillingcost), length(x.extensioncost),)
@inline number_of_model_parms(x::StaticDrillingPayoff) = length(x)

# coeficient ranges
@inline coef_range_revenue(x::StaticDrillingPayoff) =                                                        1:length(x.revenue)
@inline coef_range_drillingcost(x::StaticDrillingPayoff)  = length(x.revenue)                            .+ (1:length(x.drillingcost))
@inline coef_range_extensioncost(x::StaticDrillingPayoff) = (length(x.revenue) + length(x.drillingcost)) .+ (1:length(x.extensioncost))
@inline coef_ranges(x::StaticDrillingPayoff) = coef_range_revenue(x), coef_range_drillingcost(x), coef_range_extensioncost(x)

@inline check_coef_length(x::StaticDrillingPayoff, θ) = length(x) == length(θ) || throw(DimensionMismatch())

# flow???(
#     x::AbstractStaticPayoffs, k::Integer,             # which function
#     θ::AbstractVector, σ::T,                          # parms
#     wp::AbstractUnitProblem, i::Integer, d::Integer,  # follows sprime(wp,i,d)
#     z::Tuple, ψ::T, geoid::Real, roy::T                # other states
# )

function gradient!(f::AbstractPayoffFunction, x::AbstractVector, g::AbstractVector, args...)
    K = length(f)
    K == length(x) == length(g) || throw(DimensionMismatch())
    for k = 1:K
        g[k] = flowdθ(f,k,x, args...)
    end
end


@inline function flow(x::StaticDrillingPayoff, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::T) where {T}
    if d == 0
        @views u = flow(x.extensioncost, θ[coef_range_extensioncost(x)], σ, wp, i, d, z, ψ, geoid, roy)
    else
        @views r = flow(x.revenue,       θ[coef_range_revenue(x)],       σ, wp, i, d, z, ψ, geoid, roy)
        @views c = flow(x.drillingcost,  θ[coef_range_drillingcost(x)],  σ, wp, i, d, z, ψ, geoid, roy)
        u = r+c
    end
    return u::T
end

@inline function flowdθ(x::StaticDrillingPayoff, k::Integer, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::T)::T where {T}
    d == 0 && !_sgnext(wp,i) && return zero(T)

    kr, kc, ke = lengths(x)

    # revenue
    k < 0              && throw(DomainError(k))
    k <= kr            && return flowdθ(x.revenue,       k,       θ[coef_range_revenue(x)],       σ, wp, i, d, z, ψ, geoid, roy)
    k <= kr + kc       && return flowdθ(x.drillingcost,  k-kr,    θ[coef_range_drillingcost(x)],  σ, wp, i, d, z, ψ, geoid, roy)
    k <= kr + kc + ke  && return flowdθ(x.extensioncost, k-kr-kc, θ[coef_range_extensioncost(x)], σ, wp, i, d, z, ψ, geoid, roy)
    throw(DomainError(k))
end

@inline function flowdψ(x::StaticDrillingPayoff, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, args...)::T where {T}
    d == 0 && return flowdψ(x.extensioncost, θ, σ, wp, i, d, args...)
    r = flowdψ(x.revenue,       θ, σ, wp, i, d, args...)
    c = flowdψ(x.drillingcost,  θ, σ, wp, i, d, args...)
    return r+c
end

@inline function flowdσ(x::StaticDrillingPayoff, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, args...)::T where {T}
    d == 0 && return flowdσ(x.extensioncost, θ, σ, wp, i, d, args...)
    r = flowdσ(x.revenue,       θ, σ, wp, i, d, args...)
    c = flowdσ(x.drillingcost,  θ, σ, wp, i, d, args...)
    return r+c
end

# -------------------------------------------
# Extension
# -------------------------------------------

@inline flowdσ(::AbstractExtensionCost, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = zero(T)
@inline flowdψ(::AbstractExtensionCost, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = zero(T)

"No extension cost"
struct ExtensionCost_Zero <: AbstractExtensionCost end
length(::ExtensionCost_Zero) = 0
@inline flow(  ::ExtensionCost_Zero,             θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = zero(T)
@inline flowdθ(::ExtensionCost_Zero, k::Integer, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = nothing

"Constant extension cost"
struct ExtensionCost_Constant <: AbstractExtensionCost end
length(::ExtensionCost_Constant) = 1
@inline flow(  ::ExtensionCost_Constant,             θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = _sgnext(wp,i,d) ? θ[1]   : zero(T)
@inline flowdθ(::ExtensionCost_Constant, k::Integer, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = _sgnext(wp,i,d) ? one(T) : zero(T)

"Extension cost depends on ψ"
struct ExtensionCost_ψ <: AbstractExtensionCost end
length(::ExtensionCost_ψ) = 2
@inline flow(  ::ExtensionCost_ψ,             θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = θ[1] + θ[2]*ψ
@inline flowdθ(::ExtensionCost_ψ, k::Integer, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = k == 1 ? one(T) : ψ
@inline flowdψ(::ExtensionCost_ψ,             θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = θ[2]

# -------------------------------------------
# Drilling Cost
# -------------------------------------------

@inline flowdσ(::AbstractDrillingCost, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = zero(T)
@inline flowdψ(::AbstractDrillingCost, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = zero(T)

"Single drilling cost"
struct DrillingCost_constant <: AbstractDrillingCost end
@inline length(x::DrillingCost_constant) = 1
@inline flow(  u::DrillingCost_constant,             θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = d*θ[1]
@inline flowdθ(u::DrillingCost_constant, k::Integer, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real) where {T} = T(d)

struct DrillingCost_dgt1 <: AbstractDrillingCost end
@inline length(x::DrillingCost_dgt1) = 2
@inline flow(           u::DrillingCost_dgt1,             θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i, d, z, ψ, geoid, roy) where {T} = d*(d<=1 ? θ[1] : θ[2])
@inline function flowdθ(u::DrillingCost_dgt1, k::Integer, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i, d, z, ψ, geoid, roy) where {T}
    k == 1 && return d <= 1 ? T(d) : zero(T)
    k == 2 && return d >  1 ? T(d) : zero(T)
    throw(DomainError(k))
end


"Abstract Type for Costs w Fixed Effects"
abstract type AbstractDrillingCost_TimeFE <: AbstractDrillingCost end
@inline start(x::AbstractDrillingCost_TimeFE) = x.start
@inline stop(x::AbstractDrillingCost_TimeFE) = x.stop
@inline startstop(x::AbstractDrillingCost_TimeFE) = start(x), stop(x)
@inline time_idx(x::AbstractDrillingCost_TimeFE, t) = clamp(t, start(x), stop(x)) - start(x) + 1

"Time FE for 2008-2012"
struct DrillingCost_TimeFE <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline length(x::DrillingCost_TimeFE) = 2 + stop(x) - start(x)
@inline function flow(u::DrillingCost_TimeFE, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple{T,<:Integer}, ψ::T, geoid::Real, roy::Real)::T where {T}
    d == 1 && return    θ[time_idx(u,last(z))]
    d  > 1 && return d*(θ[time_idx(u,last(z))] + θ[length(u)])
    d  < 1 && return zero(T)
end
@inline function flowdθ(u::DrillingCost_TimeFE, k::Integer, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple{T,<:Integer}, ψ::T, geoid::Real, roy::Real)::T where {T}
    if 0 < k <= length(u)-1
        return k == time_idx(u,last(z)) ? T(d) : zero(T)
    end
    k <= length(u) && return d <= 1 ? zero(T) : T(d)
    throw(DomainError(k))
end


"Time FE w rig rates for 2008-2012"
struct DrillingCost_TimeFE_rigrate <: AbstractDrillingCost_TimeFE
    start::Int16
    stop::Int16
end
@inline length(x::DrillingCost_TimeFE_rigrate) = 3 + stop(x) - start(x)
@inline function flow(u::DrillingCost_TimeFE_rigrate, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple{T,T,<:Integer}, ψ::T, geoid::Real, roy::Real)::T where {T}
    d == 1 && return     θ[time_idx(u,last(z))] +                  θ[length(u)]*exp(z[2]  )
    d  > 1 && return d*( θ[time_idx(u,last(z))] + θ[length(u)-1] + θ[length(u)]*exp(z[2]) )
    d  < 1 && return zero(T)
end
@inline function flowdθ(u::DrillingCost_TimeFE_rigrate, k::Integer, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple{T,T,<:Integer}, ψ::T, geoid::Real, roy::Real)::T where {T}
    K = length(u)
    if 0 < k <= K-2
        return k == time_idx(u,last(z)) ? T(d) : zero(T)
    end
    k == K-1 && return d <= 1 ? zero(T) : T(d)
    k <= K   && return d == 0 ? zero(T) : d*exp(z[2])
    throw(DomainError(k))
end


# ----------------------------------------------------------------
# Drilling revenue variations
# ----------------------------------------------------------------

abstract type AbstractModelVariations end


# Whether we've constrained coefs
abstract type AbstractConstrainedType <: AbstractModelVariations end
struct Unconstrained <: AbstractConstrainedType end
struct Constrained   <: AbstractConstrainedType
    log_ogip::Float64
    α_ψ::Float64
    α_t::Float64
end
Constrained() = Constrained(STARTING_log_ogip, STARTING_α_ψ, STARTING_t)
log_ogip(x::Constrained) = x.log_ogip
α_ψ(x::Constrained) = x.α_ψ
α_t(x::Constrained) = x.α_t


# Technology
abstract type AbstractTechChange <: AbstractModelVariations end
struct NoTrend       <: AbstractTechChange  end
struct TimeTrend     <: AbstractTechChange
    baseyear::Int
end
TimeTrend() = TimeTrend(2008)
baseyear(x::TimeTrend) = x.baseyear

# taxes
abstract type AbstractTaxType <: AbstractModelVariations end
struct NoTaxes       <: AbstractTaxType end
struct WithTaxes     <: AbstractTaxType end

# ----------------------------------------------------------------
# Drilling revenue object
# ----------------------------------------------------------------

# drilling revenue
struct DrillingRevenue{Cn <: AbstractConstrainedType, Tech <: AbstractTechChange, Tax <: AbstractTaxType} <: AbstractDrillingRevenue
    constr::Cn
    tech::Tech
    tax::Tax
end

@inline log_ogip(x::DrillingRevenue{Constrained},   θ::AbstractVector) = log_ogip(x.constr)
@inline α_ψ(     x::DrillingRevenue{Constrained},   θ::AbstractVector) = α_ψ(     x.constr)
@inline α_t(     x::DrillingRevenue{Constrained},   θ::AbstractVector) = α_t(     x.constr)

@inline log_ogip(x::DrillingRevenue{Unconstrained}, θ::AbstractVector) = θ[2]
@inline α_ψ(     x::DrillingRevenue{Unconstrained}, θ::AbstractVector) = θ[3]
@inline α_t(     x::DrillingRevenue{Unconstrained}, θ::AbstractVector) = θ[4]

constrained_parms(::DrillingRevenue{<:AbstractConstrainedType, NoTrend})   = (log_ogip=2, α_ψ=3,)
constrained_parms(::DrillingRevenue{<:AbstractConstrainedType, TimeTrend}) = (log_ogip=2, α_ψ=3, α_t=4)
constrained_parms(x::StaticDrillingPayoff) = constrained_parms(x.revenue)

ConstrainedProblem(  x::AbstractPayoffComponent, args...) = x
UnconstrainedProblem(x::AbstractPayoffComponent, args...) = x
UnconstrainedProblem(x::DrillingRevenue, args...)         = DrillingRevenue(Constrained(args...), x.tech, x.tax)
ConstrainedProblem(  x::DrillingRevenue, args...)         = DrillingRevenue(Unconstrained(), x.tech, x.tax)
ConstrainedProblem(  x::StaticDrillingPayoff, args...)    = StaticDrillingPayoff(ConstrainedProblem(revenue(x), args...), ConstrainedProblem(drillingcost(x)), ConstrainedProblem(extensioncost(x)))
UnconstrainedProblem(x::StaticDrillingPayoff, args...)    = StaticDrillingPayoff(UnconstrainedProblem(revenue(x), args...), UnconstrainedProblem(drillingcost(x)), UnconstrainedProblem(extensioncost(x)))


# -------------------------------------------
# base functions
# -------------------------------------------

@inline function Eexpψ(θ4::T, σ::Number, ψ::Number, Dgt0::Bool)::T where {T}
    if Dgt0
        return θ4*ψ
    else
        ρ = _ρ(σ)
        return θ4*(ψ*ρ + θ4*0.5*(one(T)-ρ^2))
    end
end

# ----------------------------------------------------------------
# regular drilling revenue
# ----------------------------------------------------------------

@inline length(x::DrillingRevenue{Constrained}) = 1
@inline length(x::DrillingRevenue{Unconstrained, NoTrend}) = 3
@inline length(x::DrillingRevenue{Unconstrained, TimeTrend}) = 4

# ----------------------------------------------------------------
# flow revenue
# ----------------------------------------------------------------

@inline function flow(x::DrillingRevenue{Cn,NoTrend,NoTaxes}, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::Real, geoid::Real, roy::Real) where {T,Cn}
    u = d*(one(T)-roy) * exp(θ[1] + z[1] + log_ogip(x,θ)*geoid + Eexpψ(α_ψ(x,θ), σ, ψ, _Dgt0(wp,i)))
    return u::T
end

@inline function flow(x::DrillingRevenue{Cn,NoTrend,WithTaxes}, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::Real, geoid::Real, roy::Real) where {T,Cn}
    u = d*ONE_MINUS_MARGINAL_TAX_RATE*(one(T)-roy) * exp(θ[1] + log_ogip(x,θ)*geoid + Eexpψ(α_ψ(x,θ), σ, ψ, _Dgt0(wp,i))) * (exp(z[1]) - C_PER_MCF)
    return u::T
end

@inline function flow(x::DrillingRevenue{Cn,TimeTrend}, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::Real, geoid::Real, roy::Real) where {T,Cn}
    u = d*(one(T)-roy) * exp(θ[1] + z[1] + log_ogip(x,θ)*geoid + Eexpψ(α_ψ(x,θ), σ, ψ, _Dgt0(wp,i)) + α_t(x,θ)*(last(z) - baseyear(x.tech)) )
    return u::T
end

@inline function flow(x::DrillingRevenue{Cn,TimeTrend,WithTaxes}, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::Real, geoid::Real, roy::Real) where {T,Cn}
    u = d*ONE_MINUS_MARGINAL_TAX_RATE*(one(T)-roy) * exp(θ[1] + log_ogip(x,θ)*geoid +  Eexpψ(α_ψ(x,θ), σ, ψ, _Dgt0(wp,i)) + α_t(x,θ)*(last(z) - baseyear(x.tech)) ) * (exp(z[1]) - C_PER_MCF)
    return u::T
end

# ----------------------------------------------------------------
# dψ and dσ are the same across many functions
# ----------------------------------------------------------------

@inline function flowdσ(x::DrillingRevenue, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real)::T where {T}
    if !_Dgt0(wp,i) && d > 0
        return flow(x, θ, σ, wp, i, d, z, ψ, geoid, roy) * (ψ*α_ψ(x,θ) - α_ψ(x,θ)^2*_ρ(σ)) * _dρdσ(σ)
    end
    return zero(T)
end

@inline function flowdψ(x::DrillingRevenue, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real)::T where {T}
    if d > 0
        dψ = flow(x, θ, σ, wp, i, d, z, ψ, geoid, roy) *  α_ψ(x,θ)
        return _Dgt0(wp,i) ? dψ : dψ * _ρ(σ)
    end
    return zero(T)
end

# Constrained derivatives
# ------------------------------

@inline function flowdθ(x::DrillingRevenue{Unconstrained, NoTrend}, k::Integer, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real)::T where {T}
    rev = flow(x, θ, σ, wp, i, d, z, ψ, geoid, roy)
    k == 1 && return rev
    k == 2 && return rev*geoid
    k == 3 && return rev*( _Dgt0(wp,i) ? ψ : ψ*_ρ(σ) + θ[3]*(1-_ρ2(σ)))
    throw(DomainError(k))
end

@inline function flowdθ(x::DrillingRevenue{Unconstrained, TimeTrend}, k::Integer, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real)::T where {T}
    rev = flow(x, θ, σ, wp, i, d, z, ψ, geoid, roy)
    k == 1 && return rev
    k == 2 && return rev*geoid
    k == 3 && return rev*( _Dgt0(wp,i) ? ψ : ψ*_ρ(σ) + θ[3]*(1-_ρ2(σ)))
    k == 4 && return rev*( last(z) - baseyear(x.tech) )
    throw(DomainError(k))
end


# Constrained derivatives
# ------------------------------

@inline function flowdθ(x::DrillingRevenue{Constrained}, k::Integer, θ::AbstractVector{T}, σ::T, wp::AbstractUnitProblem, i::Integer, d::Integer, z::Tuple, ψ::T, geoid::Real, roy::Real)::T where {T}
    rev = flow(x, θ, σ, wp, i, d, z, ψ, geoid, roy)
    k == 1 && return rev
    throw(DomainError(k))
end
