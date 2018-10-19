export flow, flowdθ, flowdσ, flowdψ


# make primitives. Note: flow payoffs, gradient, and grad wrt `σ` must have the following structure:
# ```julia
# f(  ::Type{Val{FF}}, θ::AbstractVector{T}, σ::T,   z... , ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, roy::Real, geoid::Real)
# df( ::Type{Val{FF}}, θ::AbstractVector{T}, σ::T,   z... , ψ::T, k::Integer, d::Integer, d1::Integer, Dgt0::Bool, roy::Real, geoid::Real)
# dfσ(::Type{Val{FF}}, θ::AbstractVector{T}, σ::T,   z... , ψ::T,             d::Integer,                          roy::Real, geoid::Real)
# dfψ(::Type{Val{FF}}, θ::AbstractVector{T}, σ::T,   z... , ψ::T,             d::Integer,                          roy::Real, geoid::Real)
# ```


# functions in case we have regime-info
@inline flow(  FF::Type,  θ::AbstractVector{T}, σ::T, logp::T, regime::Integer, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T} = flow(   FF, θ, σ, logp, ψ,    d, d1, Dgt0, roy, geoid)
@inline flowrev(FF::Type, θ::AbstractVector{T}, σ::T, logp::T, regime::Integer, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T} = flowrev(FF, θ, σ, logp, ψ,    d, d1, Dgt0, roy, geoid)
@inline flowdθ(FF::Type,  θ::AbstractVector{T}, σ::T, logp::T, regime::Integer, ψ::T, k::Integer, d::Integer, d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T} = flowdθ( FF, θ, σ, logp, ψ, k, d, d1, Dgt0, roy, geoid)
@inline flowdσ(FF::Type,  θ::AbstractVector{T}, σ::T, logp::T, regime::Integer, ψ::T,             d::Integer,                          roy::T, geoid::Real) where {T} = flowdσ( FF, θ, σ, logp, ψ,    d,           roy, geoid)
@inline flowdψ(FF::Type,  θ::AbstractVector{T}, σ::T, logp::T, regime::Integer, ψ::T,             d::Integer,                          roy::T, geoid::Real) where {T} = flowdψ( FF, θ, σ, logp, ψ,    d,           roy, geoid)


function flow_extend(FF::Type, θext::AbstractVector{T}, ψ::T, roy::T, geoid::Real, d::Integer) where {T<:Real}
    d > 0 && return zero(T)
    return (θext[1] + θext[2]*ψ)::T
end


function dflow_extend(FF::Type, θext::AbstractVector{T}, ψ::T, k::Integer, roy::T, geoid::Real, d::Integer) where {T<:Real}
    d > 0  && return zero(T)
    k == 1 && return one(T)
    k == 2 && return ψ::T
    throw(error("$k out of bounds"))
end


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

function Eψ(θ4::T, σ::T, ψ::T, Dgt0::Bool) where {T<:Real}
    if Dgt0
        out = θ4*ψ
    else
        out = θ4*ψ*_ρ(σ)
    end
    return out::T
end

@inline function rev_exp(θ0::T, θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real, Dgt0::Bool, roy::Real, geoid::Real) where {T<:Real}
    r = (one(T)-θ0*roy) * exp(θ1 + θ2*logp + θ3*geoid + Eexpψ(θ4, σ, ψ, Dgt0))
    return r::T
end

@inline function rev_lin(θ0::T, θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real, Dgt0::Bool, roy::Real, geoid::Real) where {T<:Real}
    r = (one(T)-θ0*roy) * exp(θ2*logp) * (θ1 + θ3*geoid + Eψ(θ4,σ,ψ,Dgt0))
    return r::T
end

@inline drevdσ_lin(θ0::T, θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real, roy::Real, geoid::Real) where {T} = (one(T)-θ0*roy) * exp(θ2*logp) * θ4 * ψ * _dρdθρ(σ)
@inline drevdψ_lin(θ0::T, θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real, roy::Real, geoid::Real) where {T} = (one(T)-θ0*roy) * exp(θ2*logp) * θ4 * _ρ(σ)

@inline drevdσ_exp(θ0::T, θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real, roy::Real, geoid::Real) where {T} = rev_exp(θ0,θ1,θ2,θ3,θ4,σ,logp,ψ,false,roy,geoid) * (ψ*θ4 - θ4^2*_ρ(σ)) * _dρdσ(σ)
@inline drevdψ_exp(θ0::T, θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real, roy::Real, geoid::Real) where {T} = rev_exp(θ0,θ1,θ2,θ3,θ4,σ,logp,ψ,false,roy,geoid) * θ4 * _ρ(σ)

# in case we restrict the coef on royalty to be 1
@inline rev_exp(   θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real, Dgt0::Bool, roy::Real, geoid::Real) where {T} = rev_exp(   one(T), θ1, θ2, θ3, θ4, σ, logp, ψ, Dgt0, roy, geoid)
@inline rev_lin(   θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real, Dgt0::Bool, roy::Real, geoid::Real) where {T} = rev_lin(   one(T), θ1, θ2, θ3, θ4, σ, logp, ψ, Dgt0, roy, geoid)
@inline drevdσ_lin(θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real,             roy::Real, geoid::Real) where {T} = drevdσ_lin(one(T), θ1, θ2, θ3, θ4, σ, logp, ψ,       roy, geoid)
@inline drevdψ_lin(θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real,             roy::Real, geoid::Real) where {T} = drevdψ_lin(one(T), θ1, θ2, θ3, θ4, σ, logp, ψ,       roy, geoid)
@inline drevdσ_exp(θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real,             roy::Real, geoid::Real) where {T} = drevdσ_exp(one(T), θ1, θ2, θ3, θ4, σ, logp, ψ,       roy, geoid)
@inline drevdψ_exp(θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, ψ::Real,             roy::Real, geoid::Real) where {T} = drevdψ_exp(one(T), θ1, θ2, θ3, θ4, σ, logp, ψ,       roy, geoid)


# -----------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- exponential roy --------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------


@inline function flow(::Type{Val{:exproy}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    u = rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,roy,geoid) + (d==1 ?  θ[6] : θ[7] + θ[8]*d)
    d>1      && (u *= d)
    d1 == 1  && (u += θ[9])
    return u::T
end



@inline function flowdθ(::Type{Val{:exproy}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::Real, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return - convert(T,d) * exp(θ[2] + θ[3]*logp + θ[4]*geoid + Eexpψ(θ[5], σ, ψ, Dgt0) ) * roy
    k == 2  && return   convert(T,d) * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,roy,geoid)
    k == 3  && return   convert(T,d) * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,roy,geoid) * logp
    k == 4  && return   convert(T,d) * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,roy,geoid) * convert(T,geoid)
    k == 5  && return   convert(T,d) * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,roy,geoid) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    k == 6  && return  d  >  1 ? zero(T) : one(T)
    k == 7  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 8  && return  d  == 1 ? zero(T) : convert(T,d^2)
    k == 9  && return  d1 == 1 ? one(T)  : zero(T)
    throw(error("$k out of bounds"))
end

@inline flowdσ(::Type{Val{:exproy}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * drevdσ_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,roy,geoid)
@inline flowdψ(::Type{Val{:exproy}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * drevdψ_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,roy,geoid)


# -----------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- CONSTRAINED exponential roy --------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------


@inline function flow(::Type{Val{:exp1roy}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    u = rev_exp(1.0,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,roy,geoid) + (d==1 ?  θ[5] : θ[6] + θ[7]*d)
    d>1      && (u *= d)
    d1 == 1  && (u += θ[8])
    return u::T
end



@inline function flowdθ(::Type{Val{:exp1roy}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::Real, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return   convert(T,d) * rev_exp(1.0,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,roy,geoid)
    k == 2  && return   convert(T,d) * rev_exp(1.0,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,roy,geoid) * logp
    k == 3  && return   convert(T,d) * rev_exp(1.0,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,roy,geoid) * convert(T,geoid)
    k == 4  && return   convert(T,d) * rev_exp(1.0,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,roy,geoid) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    k == 5  && return  d  >  1 ? zero(T) : one(T)
    k == 6  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 7  && return  d  == 1 ? zero(T) : convert(T,d^2)
    k == 8  && return  d1 == 1 ? one(T)  : zero(T)
    throw(error("$k out of bounds"))
end

@inline flowdσ(::Type{Val{:exp1roy}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * drevdσ_exp(1.0,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,roy,geoid)
@inline flowdψ(::Type{Val{:exp1roy}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * drevdψ_exp(1.0,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,roy,geoid)

# -----------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- DOUBLE CONSTRAINED exponential roy --------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------


@inline function flow(::Type{Val{:exp1roy1p}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
    d == 0 && return zero(T)
    u = rev_exp(1.0,θ[1],1.0,θ[2],θ[3],σ,logp,ψ,Dgt0,roy,geoid) + (d==1 ?  θ[4] : θ[5] + θ[6]*d)
    d>1      && (u *= d)
    d1 == 1  && (u += θ[7])
    return u::T
end



@inline function flowdθ(::Type{Val{:exp1roy1p}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::Real, geoid::Real) where {T}
    d == 0  && return zero(T)

    k == 1  && return   convert(T,d) * rev_exp(1.0,θ[1],1.0,θ[2],θ[3],σ,logp,ψ,Dgt0,roy,geoid)
    k == 2  && return   convert(T,d) * rev_exp(1.0,θ[1],1.0,θ[2],θ[3],σ,logp,ψ,Dgt0,roy,geoid) * convert(T,geoid)
    k == 3  && return   convert(T,d) * rev_exp(1.0,θ[1],1.0,θ[2],θ[3],σ,logp,ψ,Dgt0,roy,geoid) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    k == 4  && return  d  >  1 ? zero(T) : one(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 6  && return  d  == 1 ? zero(T) : convert(T,d^2)
    k == 7  && return  d1 == 1 ? one(T)  : zero(T)
    throw(error("$k out of bounds"))
end

@inline flowdσ(::Type{Val{:exp1roy1p}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * drevdσ_exp(1.0,θ[1],1.0,θ[2],θ[3],σ,logp,ψ,roy,geoid)
@inline flowdψ(::Type{Val{:exp1roy1p}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * drevdψ_exp(1.0,θ[1],1.0,θ[2],θ[3],σ,logp,ψ,roy,geoid)

# # -----------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------- linear ------------------------------------------------------------------
# # -----------------------------------------------------------------------------------------------------------------------------
#
# @inline function flow(::Type{Val{:lin}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::Real, geoid::Real) where {T}
#     d == 0 && return zero(T)
#     u = rev_lin(θ[1], θ[2], θ[3], θ[4], σ, logp, ψ, Dgt0, roy, geoid) + (d==1 ? θ[5] : θ[6])
#     d>1      && (u *= convert(T,d))
#     d1 == 1  && (u += θ[7])
#     return u::T
# end
#
# @inline function flowdθ(::Type{Val{:lin}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
#     d == 0  && return zero(T)
#
#     k == 1  && return  convert(T,d) * (one(T)-roy) * exp(θ[2]*logp)
#     k == 2  && return  convert(T,d)                                 * logp  * rev_lin(θ[1], θ[2], θ[3], θ[4], σ, logp, ψ, Dgt0, roy, geoid)
#     k == 3  && return  convert(T,d) * (one(T)-roy) * exp(θ[2]*logp) * geoid
#     k == 4  && return  convert(T,d) * (one(T)-roy) * exp(θ[2]*logp) * (Dgt0 ? ψ : ψ * _ρ(σ))
#
#     k == 5  && return  d  >  1 ? zero(T) : one(T)
#     k == 6  && return  d  == 1 ? zero(T) : convert(T,d)
#     k == 7  && return  d1 == 1 ? one(T)  : zero(T)
#     throw(error("$k out of bounds"))
# end
#
# @inline flowdσ(::Type{Val{:lin}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * drevdσ_lin(θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,roy,geoid)
# @inline flowdψ(::Type{Val{:lin}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = d == 0 ? zero(T) : convert(T,d) * drevdψ_lin(θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,roy,geoid)
#
# # -----------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------- BREAK exponential --------------------------------------------------------------------
# # -----------------------------------------------------------------------------------------------------------------------------
#
# function flow(::Type{Val{:breakexp}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::Real, geoid::Real) where {T}
#     d == 0 && return zero(T)
#     if !Dgt0
#         u = rev_exp(θ[1], θ[2], θ[3], θ[4], σ, logp, ψ, Dgt0, roy, geoid) + (d==1 ? θ[5] : θ[6])
#     else
#         u = rev_exp(θ[7], θ[8], θ[9], θ[10], σ, logp, ψ, Dgt0, roy, geoid) + (d==1 ? θ[11] : θ[12])
#     end
#     d>1      && (u *= d)
#     d1 == 1  && (u += θ[13])
#     return u::T
# end
#
# function flowdθ(::Type{Val{:breakexp}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::Real, geoid::Real) where {T}
#     d == 0  && return zero(T)
#     k == 1  && return   Dgt0           ? zero(T) : convert(T,d) * rev_exp(θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,roy,geoid)
#     k == 2  && return   Dgt0           ? zero(T) : convert(T,d) * rev_exp(θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,roy,geoid) * logp
#     k == 3  && return   Dgt0           ? zero(T) : convert(T,d) * rev_exp(θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,roy,geoid) * convert(T,geoid)
#     k == 4  && return   Dgt0           ? zero(T) : convert(T,d) * rev_exp(θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,roy,geoid) * (ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))
#     k == 5  && return   Dgt0 || d > 1  ? zero(T) : one(T)
#     k == 6  && return   Dgt0 || d == 1 ? zero(T) : convert(T,d)
#
#     k == 7  && return  !Dgt0           ? zero(T) : convert(T,d) * rev_exp(θ[7],θ[8],θ[9],θ[10],σ,logp,ψ,Dgt0,roy,geoid)
#     k == 8  && return  !Dgt0           ? zero(T) : convert(T,d) * rev_exp(θ[7],θ[8],θ[9],θ[10],σ,logp,ψ,Dgt0,roy,geoid) * logp
#     k == 9  && return  !Dgt0           ? zero(T) : convert(T,d) * rev_exp(θ[7],θ[8],θ[9],θ[10],σ,logp,ψ,Dgt0,roy,geoid) * convert(T,geoid)
#     k == 10 && return  !Dgt0           ? zero(T) : convert(T,d) * rev_exp(θ[7],θ[8],θ[9],θ[10],σ,logp,ψ,Dgt0,roy,geoid) * ψ
#     k == 11 && return  !Dgt0 || d >  1 ? zero(T) : one(T)
#     k == 12 && return  !Dgt0 || d == 1 ? zero(T) : convert(T,d)
#
#     k == 13 && return  d1 == 1 ? one(T)  : zero(T)
#     throw(error("$k out of bounds"))
# end
#
# @inline flowdσ(::Type{Val{:breakexp}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::Real, geoid::Real) where {T} = flowdσ(Val{:exp}, θ, σ, logp, ψ, d, roy, geoid)
# @inline flowdψ(::Type{Val{:breakexp}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::Real, geoid::Real) where {T} = flowdψ(Val{:exp}, θ, σ, logp, ψ, d, roy, geoid)
#
# # -----------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------- BREAK linear ------------------------------------------------------------------
# # -----------------------------------------------------------------------------------------------------------------------------
#
# @inline function flow(::Type{Val{:breaklin}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer,             d1::Integer, Dgt0::Bool, roy::Real, geoid::Real) where {T}
#     d == 0 && return zero(T)
#     if !Dgt0
#         u = rev_lin(θ[1], θ[2], θ[3], θ[4], σ, logp, ψ, Dgt0, roy, geoid) + (d==1 ? θ[5] : θ[6])
#     else
#         u = rev_lin(θ[7], θ[8], θ[9], θ[10], σ, logp, ψ, Dgt0, roy, geoid) + (d==1 ? θ[11] : θ[12])
#     end
#     d>1      && (u *= convert(T,d))
#     d1 == 1  && (u += θ[13])
#     return u::T
# end
#
# @inline function flowdθ(::Type{Val{:breaklin}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, roy::T, geoid::Real) where {T}
#     d == 0  && return zero(T)
#
#     k == 1  && return   Dgt0           ? zero(T) : convert(T,d) * (one(T)-roy) * exp(θ[2]*logp)
#     k == 2  && return   Dgt0           ? zero(T) : convert(T,d)                                 * logp  * rev_lin(θ[1], θ[2], θ[3], θ[4], σ, logp, ψ, false, roy, geoid)
#     k == 3  && return   Dgt0           ? zero(T) : convert(T,d) * (one(T)-roy) * exp(θ[2]*logp) * convert(T,geoid)
#     k == 4  && return   Dgt0           ? zero(T) : convert(T,d) * (one(T)-roy) * exp(θ[2]*logp) * ψ * _ρ(σ)
#     k == 5  && return   Dgt0 || d > 1  ? zero(T) : one(T)
#     k == 6  && return   Dgt0 || d == 1 ? zero(T) : convert(T,d)
#
#     k == 7  && return  !Dgt0           ? zero(T) : convert(T,d) * (one(T)-roy) * exp(θ[8]*logp)
#     k == 8  && return  !Dgt0           ? zero(T) : convert(T,d)                                 * logp  * rev_lin(θ[7], θ[8], θ[9], θ[10], σ, logp, ψ, true, roy, geoid)
#     k == 9  && return  !Dgt0           ? zero(T) : convert(T,d) * (one(T)-roy) * exp(θ[8]*logp) * convert(T,geoid)
#     k == 10 && return  !Dgt0           ? zero(T) : convert(T,d) * (one(T)-roy) * exp(θ[8]*logp) * ψ
#     k == 11 && return  !Dgt0 || d >  1 ? zero(T) : one(T)
#     k == 12 && return  !Dgt0 || d == 1 ? zero(T) : convert(T,d)
#
#     k == 13  && return  d1 == 1 ? one(T)  : zero(T)
#     throw(error("$k out of bounds"))
# end
#
# @inline flowdσ(::Type{Val{:breaklin}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = flowdσ(Val{:lin},θ,σ,logp,ψ,d,roy,geoid)
# @inline flowdψ(::Type{Val{:breaklin}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, roy::T, geoid::Real) where {T} = flowdψ(Val{:lin},θ,σ,logp,ψ,d,roy,geoid)

# # -----------------------------------------------------------------------------------------------------------------------------
# # --------------------------------------------------- Old ------------------------------------------------------------------
# # -----------------------------------------------------------------------------------------------------------------------------
#
# # In case we have regimes
# @inline rev_exp(   θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, regime::Integer ψ::Real, Dgt0::Bool, roy::Real, geoid::Real) where {T} = rev_exp(   θ1,θ2,θ3,θ4,σ,logp,ψ,Dgt0,roy,geoid)
# @inline rev_lin(   θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, regime::Integer ψ::Real, Dgt0::Bool, roy::Real, geoid::Real) where {T} = rev_lin(   θ1,θ2,θ3,θ4,σ,logp,ψ,Dgt0,roy,geoid)
#
# @inline drevdσ_lin(θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, regime::Integer,ψ::Real,             roy::Real, geoid::Real) where {T} = drevdσ_lin(θ1,θ2,θ3,θ4,σ,logp,ψ,     roy,geoid)
# @inline drevdψ_lin(θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, regime::Integer,ψ::Real,             roy::Real, geoid::Real) where {T} = drevdψ_lin(θ1,θ2,θ3,θ4,σ,logp,ψ,     roy,geoid)
#
# @inline drevdσ_exp(θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, regime::Integer,ψ::Real,             roy::Real, geoid::Real) where {T} = drevdσ_exp(θ1,θ2,θ3,θ4,σ,logp,ψ,     roy,geoid)
# @inline drevdψ_exp(θ1::T, θ2::T, θ3::T, θ4::T, σ::T, logp::Real, regime::Integer,ψ::Real,             roy::Real, geoid::Real) where {T} = drevdψ_exp(θ1,θ2,θ3,θ4,σ,logp,ψ,     roy,geoid)
