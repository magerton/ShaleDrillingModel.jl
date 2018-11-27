export flow, flowdθ, flowdσ, flowdψ

# functions in case we have volatility regime
@inline flowrev(FF::Type, θ::AbstractVector{T}, σ::T, logp::T, logvol::Real, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool,                geoid::Real, roy::T) where {T} = flowrev(FF, θ, σ, logp, ψ,    d, d1, Dgt0,                  geoid, roy)
@inline flowdσ( FF::Type, θ::AbstractVector{T}, σ::T, logp::T, logvol::Real, ψ::T,             d::Integer,                                         geoid::Real, roy::T) where {T} = flowdσ( FF, θ, σ, logp, ψ,    d,                            geoid, roy)
@inline flow(   FF::Type, θ::AbstractVector{T}, σ::T, logp::T, logvol::Real, ψ::T,             d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {T} = flow(   FF, θ, σ, logp, ψ,    d, d1, Dgt0, sgn_ext,         geoid, roy)
@inline flowdθ( FF::Type, θ::AbstractVector{T}, σ::T, logp::T, logvol::Real, ψ::T, k::Integer, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {T} = flowdθ( FF, θ, σ, logp, ψ, k, d, d1, Dgt0, sgn_ext,         geoid, roy)
@inline flowdψ( FF::Type, θ::AbstractVector{T}, σ::T, logp::T, logvol::Real, ψ::T,             d::Integer, st::Union{state,Bool},                  geoid::Real, roy::T) where {T} = flowdψ( FF, θ, σ, logp, ψ,    d, st,                        geoid, roy)
@inline flowdψ( FF::Type, θ::AbstractVector{T}, σ::T, logp::T,               ψ::T,             d::Integer, st::state,                              geoid::Real, roy::T) where {T} = flowdψ( FF, θ, σ, logp, ψ,    d, _sign_lease_extension(st), geoid, roy)

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

# -----------------------------------------
# Flows
# -----------------------------------------


@inline function flow(::Type{Val{:exproy}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {T}
    if d == 0
        sgn_ext && return θ[9]
        return zero(T)
    end
    u = rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy) + (d==1 ?  θ[6] : θ[7] ) # + θ[8]*d)
    d>1      && (u *= d)
    d1 == 1  && (u += θ[8])
    return u::T
end


@inline function flow(::Type{Val{:exp}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {T}
    if d == 0
        sgn_ext && return θ[8]
        return zero(T)
    end
    u = rev_exp(1,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,geoid, roy) + (d==1 ?  θ[5] : θ[6] ) # + θ[8]*d)
    d>1      && (u *= d)
    d1 == 1  && (u += θ[7])
    return u::T
end


@inline function flow(::Type{Val{:constr}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {T}
    if d == 0
        sgn_ext && return θ[7]
        return zero(T)
    end
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid, roy) + (d==1 ?  θ[4] : θ[5] ) # + θ[8]*d)
    d>1      && (u *= d)
    d1 == 1  && (u += θ[6])
    return u::T
end


@inline function flow(::Type{Val{:constr_onecost}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {T}
    if d == 0
        sgn_ext && return θ[5]
        return zero(T)
    end
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid, roy) + θ[4]
    d>1      && (u *= d)
    return u::T
end



@inline function flow(::Type{Val{:exproy_Dgt0}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {T}
    if d == 0
        sgn_ext && return θ[9]
        return zero(T)
    end
    u = rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy) + (!Dgt0 ? θ[6] : θ[7] ) # + θ[8]*d)
    d>1      && (u *= d)
    d1 == 1  && (u += θ[8])
    return u::T
end




@inline function flow(::Type{Val{:exproy_extend}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {T}
    if d == 0
        sgn_ext && return θ[9] + θ[10]*exp(ψ)
        return zero(T)
    end
    u = rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy) + (d==1 ?  θ[6] : θ[7] ) # + θ[8]*d)
    d>1      && (u *= d)
    d1 == 1  && (u += θ[8])
    return u::T
end

@inline function flow(::Type{Val{:exproy_extend_constr}}, θ::AbstractVector{T}, σ::T,    logp::T, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {T}
    if d == 0
        sgn_ext && return θ[7] + θ[8]*exp(ψ)
        return zero(T)
    end
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid, roy) + (d==1 ?  θ[4] : θ[5] ) # + θ[8]*d)
    d>1      && (u *= d)
    d1 == 1  && (u += θ[6])
    return u::T
end


# -----------------------------------------
# dθ
# -----------------------------------------


@inline function flowdθ(::Type{Val{:exproy_Dgt0}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return - d * exp(θ[2] + θ[3]*logp + θ[4]*geoid + Eexpψ(θ[5], σ, ψ, Dgt0) ) * roy
    k == 2  && return   d * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy)
    k == 3  && return   d * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy) * logp
    k == 4  && return   d * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy) * convert(T,geoid)
    k == 5  && return   d * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 6  && return  !Dgt0   ? convert(T,d) : zero(T)
    k == 7  && return  !Dgt0   ? zero(T)      : convert(T,d)
    k == 8  && return  d1 == 1 ? one(T)       : zero(T)

    # extension cost
    k == 9  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:exproy}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return - d * exp(θ[2] + θ[3]*logp + θ[4]*geoid + Eexpψ(θ[5], σ, ψ, Dgt0) ) * roy
    k == 2  && return   d * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy)
    k == 3  && return   d * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy) * logp
    k == 4  && return   d * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy) * convert(T,geoid)
    k == 5  && return   d * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 6  && return  d  == 1 ? one(T)  : zero(T)
    k == 7  && return  d  == 1 ? zero(T) : convert(T,d)
    # k == 8  && return  d  == 1 ? zero(T) : convert(T,d^2)
    k == 8  && return  d1 == 1 ? one(T)  : zero(T)

    # extension cost
    k == 9  && return d == 0 && sgn_ext ? one(T) : zero(T)
    # k == 10 && return d == 0 && sgn_ext ? exp(ψ) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:exp}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,geoid, roy)
    k == 2  && return   d * rev_exp(1,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,geoid, roy) * logp
    k == 3  && return   d * rev_exp(1,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,geoid, roy) * convert(T,geoid)
    k == 4  && return   d * rev_exp(1,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,Dgt0,geoid, roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 5  && return  d  == 1 ? one(T)  : zero(T)
    k == 6  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 7  && return  d1 == 1 ? one(T)  : zero(T)

    # extension cost
    k == 8  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:constr}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 4  && return  d  == 1 ? one(T)  : zero(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 6  && return  d1 == 1 ? one(T)  : zero(T)

    # extension cost
    k == 7  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:constr_onecost}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 4  && return d == 0 ? zero(T) : convert(T,d)

    # extension cost
    k == 5  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end






@inline function flowdθ(::Type{Val{:exproy_extend}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return - d * exp(θ[2] + θ[3]*logp + θ[4]*geoid + Eexpψ(θ[5], σ, ψ, Dgt0) ) * roy
    k == 2  && return   d * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy)
    k == 3  && return   d * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy) * logp
    k == 4  && return   d * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy) * convert(T,geoid)
    k == 5  && return   d * rev_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,Dgt0,geoid, roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 6  && return  d  == 1 ? one(T)  : zero(T)
    k == 7  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 8  && return  d1 == 1 ? one(T)  : zero(T)

    # extension cost
    k == 9  && return d == 0 && sgn_ext ? one(T) : zero(T)
    k == 10 && return d == 0 && sgn_ext ? exp(ψ) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:exproy_extend_constr}}, θ::AbstractVector{T}, σ::T,     logp::T, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid, roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid, roy) * convert(T,geoid)
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid, roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 4  && return  d  == 1 ? one(T)  : zero(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 6  && return  d1 == 1 ? one(T)  : zero(T)

    # extension cost
    k == 7  && return d == 0 && sgn_ext ? one(T) : zero(T)
    k == 8  && return d == 0 && sgn_ext ? exp(ψ) : zero(T)

    throw(error("$k out of bounds"))
end



# -----------------------------------------
# dσ
# -----------------------------------------

@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{Type{Val{:exproy}}, Type{Val{:exproy_extend}}, Type{Val{:exproy_Dgt0}}}, T}
    if d == 0
        return zero(T)
    else
        return d * drevdσ_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,geoid, roy)
    end
end

@inline function flowdσ(::FF, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, geoid::Real, roy::T)::T where {FF <: Union{ Type{Val{:exproy_extend_constr}}, Type{Val{:constr}}, Type{Val{:constr_onecost}} }, T}
    if d == 0
        return zero(T)
    else
        return d * drevdσ_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,geoid, roy)
    end
end

@inline function flowdσ(::Type{Val{:exp}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, geoid::Real, roy::T)::T where {T}
    if d == 0
        return zero(T)
    else
        return d * drevdσ_exp(1,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,geoid, roy)
    end
end


# -----------------------------------------
# dψ
# -----------------------------------------


@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {T, FF <: Union{Type{Val{:exproy}}, Type{Val{:exproy_Dgt0}}}}
    if d == 0
        return zero(T) # sgn_ext ? θ[10] : zero(T)
    else
        return (d * drevdψ_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,geoid, roy))::T
    end
end



@inline function flowdψ(::Type{Val{:exp}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {T}
    if d == 0
        return zero(T) # sgn_ext ? θ[10] : zero(T)
    else
        return (d * drevdψ_exp(1,θ[1],θ[2],θ[3],θ[4],σ,logp,ψ,geoid, roy))::T
    end
end


@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:constr}}, Type{Val{:constr_onecost}} }, T}
    if d == 0
        return zero(T) # sgn_ext ? θ[10] : zero(T)
    else
        return (d * drevdψ_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,geoid, roy))::T
    end
end




@inline function flowdψ(::Type{Val{:exproy_extend}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T)::T where {T}
    if d == 0
        return sgn_ext ? θ[10]*exp(ψ) : zero(T)
    else
        return d * drevdψ_exp(θ[1],θ[2],θ[3],θ[4],θ[5],σ,logp,ψ,geoid, roy)
    end
end


@inline function flowdψ(::Type{Val{:exproy_extend_constr}}, θ::AbstractVector{T}, σ::T, logp::T, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T)::T where {T}
    if d == 0
        return sgn_ext ? θ[8]*exp(ψ) : zero(T)
    else
        return d * drevdψ_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,geoid, roy)
    end
end
