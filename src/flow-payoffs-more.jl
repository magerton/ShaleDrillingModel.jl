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


# -----------------------------------------------------
# -----------------------------------------------------


@inline function flow(::Type{Val{:one}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[5]
        return zero(T)
    end
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid, roy) + θ[4]
    d>1      && (u *= d)
    # d1 == 1  && (u += θ[6])
    return u::T
end

@inline function flow(::Type{Val{:dgt1}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[6]
        return zero(T)
    end
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid, roy) + (d==1 ? θ[4] : θ[5] )
    d>1      && (u *= d)
    return u::T
end



@inline function flow(::Type{Val{:dgt1_cost}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[7]
        return zero(T)
    end
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid, roy) + (d==1 ? θ[4] : θ[5] ) + θ[6]*exp(z[2])
    d>1      && (u *= d)
    return u::T
end


@inline function flow(::Type{Val{:dgt1_pricecost}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[8]
        return zero(T)
    end
    u = rev_exp(1,θ[1],θ[4],θ[2],θ[3],σ,first(z),ψ,Dgt0,geoid, roy) + (d==1 ? θ[5] : θ[6] ) + θ[7]*exp(z[2])
    d>1      && (u *= d)
    return u::T
end


@inline function flow(::Type{Val{:dgt1_cost_Dgt0}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[7]
        return zero(T)
    end
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid, roy) + (d==1 ? θ[4] : θ[5] ) + θ[6]*exp(z[2])
    d>1      && (u *= d)
    !Dgt0 && d > 0 && (u += θ[8])
    return u::T
end



@inline function flow(::Type{Val{:dgt1_d1}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[7]
        return zero(T)
    end
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid, roy) + (d==1 ? θ[4] : θ[5] ) # + θ[8]*d)
    d>1      && (u *= d)
    d1 == 1  && (u += θ[6])
    return u::T
end


@inline function flow(::Type{Val{:Dgt0}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[6]
        return zero(T)
    end
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid, roy) + (!Dgt0 ?  θ[4] : θ[5] )
    d>1      && (u *= d)
    return u::T
end

@inline function flow(::Type{Val{:dgt1_ext}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T},  ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[6] + θ[7]*ψ
        return zero(T)
    end
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid, roy) + (d==1 ?  θ[4] : θ[5] )
    d>1      && (u *= d)
    return u::T
end

# --------------------

@inline function flow(::Type{Val{:one_restr}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[3]
        return zero(T)
    end
    u = rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy) + θ[2]
    d>1      && (u *= d)
    return u::T
end


@inline function flow(::Type{Val{:dgt1_restr}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[4]
        return zero(T)
    end
    u = rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy) + (d==1 ? θ[2] : θ[3] )
    d>1      && (u *= d)
    return u::T
end

@inline function flow(::Type{Val{:dgt1_cost_restr}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[5]
        return zero(T)
    end
    u = rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy) + (d==1 ? θ[2] : θ[3] ) + θ[4]*exp(z[2])
    d>1      && (u *= d)
    return u::T
end


@inline function flow(::Type{Val{:dgt1_pricecost_restr}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[6]
        return zero(T)
    end
    u = rev_exp(1, θ[1], θ[2], STARTING_log_ogip, STARTING_σ_ψ, σ, first(z), ψ, Dgt0, geoid, roy) + (d==1 ? θ[3] : θ[4] ) + θ[5]*exp(z[2])
    d>1      && (u *= d)
    return u::T
end




@inline function flow(::Type{Val{:dgt1_cost_Dgt0_restr}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[5]
        return zero(T)
    end
    u = rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy) + (d==1 ? θ[2] : θ[3] ) + θ[4]*exp(z[2])
    d>1      && (u *= d)
    !Dgt0 && d > 0 && (u += θ[6])
    return u::T
end




@inline function flow(::Type{Val{:dgt1_d1_restr}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T},  ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[5]
        return zero(T)
    end
    u = rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy) + (d==1 ? θ[2] : θ[3] )
    d>1      && (u *= d)
    d1 == 1  && (u += θ[4])
    return u::T
end



@inline function flow(::Type{Val{:dgt1_ext_restr}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[4] + θ[5]*ψ
        return zero(T)
    end
    u = rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy) + (d==1 ? θ[2] : θ[3] )
    d>1      && (u *= d)
    return u::T
end


@inline function flow(::Type{Val{:Dgt0_restr}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[4]
        return zero(T)
    end
    u = rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy) + (!Dgt0 ? θ[2] : θ[3] )
    d>1      && (u *= d)
    return u::T
end


# -----------------------------------------
# dθ
# -----------------------------------------

@inline function flowdθ(::Type{Val{:one}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 4  && return d == 0 ? zero(T) : convert(T,d)

    # extension cost
    k == 5  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:dgt1}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T},  ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    k == 4  && return  d  == 1 ? one(T)  : zero(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    # k == 6  && return  d1 == 1 ? one(T)  : zero(T)

    k == 6  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:dgt1_cost}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T},  ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    k == 4  && return  d  == 1 ? one(T)  : zero(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 6  && return  d*exp(z[2])

    k == 7  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:dgt1_pricecost}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T},  ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return   d * rev_exp(1,θ[1],θ[4],θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],θ[4],θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],θ[4],θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))
    k == 4  && return   d * rev_exp(1,θ[1],θ[4],θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * first(z)

    k == 5  && return  d  == 1 ? one(T)  : zero(T)
    k == 6  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 7  && return  d*exp(z[2])

    k == 8  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:dgt1_cost_Dgt0}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T},  ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    k == 4  && return  d  == 1 ? one(T)  : zero(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 6  && return  d*exp(z[2])

    k == 7  && return d == 0 && sgn_ext ? one(T) : zero(T)
    k == 8  && return !Dgt0 && d > 0 ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:dgt1_d1}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T},  ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    k == 4  && return  d  == 1 ? one(T)  : zero(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 6  && return  d1 == 1 ? one(T)  : zero(T)

    k == 7  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:dgt1_ext}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    k == 4  && return  d  == 1 ? one(T)  : zero(T)
    k == 5  && return  d  == 1 ? zero(T) : convert(T,d)
    # k == 6  && return  d1 == 1 ? one(T)  : zero(T)

    k == 6  && return d == 0 && sgn_ext ? one(T) : zero(T)
    k == 7  && return d == 0 && sgn_ext ? ψ : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:Dgt0}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    k == 4  && return  !Dgt0 ? convert(T,d) : zero(T)
    k == 5  && return  !Dgt0 ? zero(T)      : convert(T,d)
    # k == 6  && return  d1 == 1 ? one(T)  : zero(T)

    k == 6  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end





@inline function flowdθ(::Type{Val{:one_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    k == 2  && return d == 0 ? zero(T) : convert(T,d)

    k == 3  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:dgt1_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    k == 2  && return d == 1 ? one(T)  : zero(T)
    k == 3  && return d == 1 ? zero(T) : convert(T,d)

    k == 4  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:dgt1_cost_restr}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return   d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    k == 2  && return  d  == 1 ? one(T)  : zero(T)
    k == 3  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 4  && return  d*exp(z[2])

    k == 5  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:dgt1_pricecost_restr}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T},  ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return   d * rev_exp(1, θ[1], θ[2], STARTING_log_ogip, STARTING_σ_ψ, σ, first(z), ψ, Dgt0, geoid, roy)
    k == 2  && return   d * rev_exp(1, θ[1], θ[2], STARTING_log_ogip, STARTING_σ_ψ, σ, first(z), ψ, Dgt0, geoid, roy) * first(z)

    k == 3  && return  d  == 1 ? one(T)  : zero(T)
    k == 4  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 5  && return  d*exp(z[2])

    k == 6  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end




@inline function flowdθ(::Type{Val{:dgt1_cost_Dgt0_restr}}, θ::AbstractVector{T}, σ::T,    z::NTuple{N,T},  ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return   d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    k == 2  && return  d  == 1 ? one(T)  : zero(T)
    k == 3  && return  d  == 1 ? zero(T) : convert(T,d)
    k == 4  && return  d*exp(z[2])

    k == 5  && return d == 0 && sgn_ext ? one(T) : zero(T)
    k == 6  && return !Dgt0 && d > 0 ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:dgt1_d1_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    k == 2  && return d  == 1 ? one(T)  : zero(T)
    k == 3  && return d  == 1 ? zero(T) : convert(T,d)
    k == 4  && return d1 == 1 ? one(T)  : zero(T)

    k == 5  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:dgt1_ext_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    k == 2  && return d == 1 ? one(T)  : zero(T)
    k == 3  && return d == 1 ? zero(T) : convert(T,d)

    k == 4  && return d == 0 && sgn_ext ? one(T) : zero(T)
    k == 5  && return d == 0 && sgn_ext ? ψ : zero(T)

    throw(error("$k out of bounds"))
end


@inline function flowdθ(::Type{Val{:Dgt0_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T},  ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    k == 1  && return d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    k == 2  && return !Dgt0 ? convert(T,d) : zero(T)
    k == 3  && return !Dgt0 ? zero(T)      : convert(T,d)

    k == 4  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end


# -----------------------------------------
# dσ
# -----------------------------------------



# -----------------------------------------
# dψ
# -----------------------------------------




@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:dgt1_ext}} },N, T}
    d == 0  && return sgn_ext ? θ[7] : zero(T)
    return (d * drevdψ_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,geoid,roy))::T
end

@inline function flowdψ(::FF, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, sgn_ext::Bool, geoid::Real, roy::T) where {FF <: Union{ Type{Val{:dgt1_ext_restr}} },N, T}
    d == 0  && return sgn_ext ? θ[5] : zero(T)
    return (d * drevdψ_exp_restricted(θ[1],σ,z[1],ψ,geoid,roy))::T
end


















# -----------------------------------------
# Allow price to vary
# -----------------------------------------














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















# ---------------------------------------------------------




@inline function flow(::Type{Val{:cheb2}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[7]
        return zero(T)
    end
    x = last(z)
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,first(z),ψ,Dgt0,geoid, roy) + θ[4]*cheb0(x) + θ[5]*cheb1(x) + θ[6]*cheb2(x)
    d>1 && (u *= d)
    return u::T
end

@inline function flow(::Type{Val{:cheb2_restr}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[5]
        return zero(T)
    end
    x = last(z)
    u = rev_exp_restricted(θ[1], σ, first(z), ψ, Dgt0, geoid, roy) + θ[2]*cheb0(x) + θ[3]*cheb1(x) + θ[4]*cheb2(x)
    d>1 && (u *= d)
    return u::T
end



@inline function flow(::Type{Val{:cheb3}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[8]
        return zero(T)
    end
    x = last(z)
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,first(z),ψ,Dgt0,geoid, roy) + θ[4]*cheb0(x) + θ[5]*cheb1(x) + θ[6]*cheb2(x) + θ[7]*cheb3(x)
    d>1 && (u *= d)
    return u::T
end

@inline function flow(::Type{Val{:cheb3_restr}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[6]
        return zero(T)
    end
    x = last(z)
    u = rev_exp_restricted(θ[1], σ, first(z), ψ, Dgt0, geoid, roy) + θ[2]*cheb0(x) + θ[3]*cheb1(x) + θ[4]*cheb2(x) + θ[5]*cheb3(x)
    d>1 && (u *= d)
    return u::T
end



@inline function flow(::Type{Val{:cheb3_dgt1}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[9]
        return zero(T)
    end
    x = last(z)
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,first(z),ψ,Dgt0,geoid, roy) + (d==1 ? θ[4] : θ[5] )*cheb0(x) + θ[6]*cheb1(x) + θ[7]*cheb2(x) + θ[8]*cheb3(x)
    d>1 && (u *= d)
    return u::T
end

@inline function flow(::Type{Val{:cheb3_dgt1_restr}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[7]
        return zero(T)
    end
    x = last(z)
    u = rev_exp_restricted(θ[1], σ, first(z), ψ, Dgt0, geoid, roy) + (d==1 ? θ[2] : θ[3] )*cheb0(x) + θ[4]*cheb1(x) + θ[5]*cheb2(x) + θ[6]*cheb3(x)
    d>1 && (u *= d)
    return u::T
end


@inline function flow(::Type{Val{:cheb3_cost}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[10]
        return zero(T)
    end
    logp, logc, t = z
    u = rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid, roy) + (d==1 ? θ[4] : θ[5] )*cheb0(t) + θ[6]*cheb1(t) + θ[7]*cheb2(t) + θ[8]*cheb3(t) + θ[9]*exp(logc)
    d>1 && (u *= d)
    return u::T
end

@inline function flow(::Type{Val{:cheb3_cost_restr}}, θ::AbstractVector{T}, σ::T, z::NTuple{N,T}, ψ::T, d::Integer, d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T) where {N,T}
    if d == 0
        sgn_ext && return θ[8]
        return zero(T)
    end
    logp, logc, t = z
    u = rev_exp_restricted(θ[1], σ, logp, ψ, Dgt0, geoid, roy) + (d==1 ? θ[2] : θ[3] )*cheb0(t) + θ[4]*cheb1(t) + θ[5]*cheb2(t) + θ[6]*cheb3(t) + θ[7]*exp(logc)
    d>1 && (u *= d)
    return u::T
end


















@inline function flowdθ(::Type{Val{:cheb2}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 4  && return d == 0 ? zero(T) : T( d*cheb0(z[2]) )
    k == 5  && return d == 0 ? zero(T) : T( d*cheb1(z[2]) )
    k == 6  && return d == 0 ? zero(T) : T( d*cheb2(z[2]) )

    # extension cost
    k == 7  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:cheb2_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    # drilling cost
    k == 2  && return d == 0 ? zero(T) : T( d*cheb0(z[2]) )
    k == 3  && return d == 0 ? zero(T) : T( d*cheb1(z[2]) )
    k == 4  && return d == 0 ? zero(T) : T( d*cheb2(z[2]) )

    # extension cost
    k == 5  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end







@inline function flowdθ(::Type{Val{:cheb3}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 4  && return d == 0 ? zero(T) : T( d*cheb0(z[2]) )
    k == 5  && return d == 0 ? zero(T) : T( d*cheb1(z[2]) )
    k == 6  && return d == 0 ? zero(T) : T( d*cheb2(z[2]) )
    k == 7  && return d == 0 ? zero(T) : T( d*cheb3(z[2]) )

    # extension cost
    k == 8  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:cheb3_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    # drilling cost
    k == 2  && return d == 0 ? zero(T) : T( d*cheb0(z[2]) )
    k == 3  && return d == 0 ? zero(T) : T( d*cheb1(z[2]) )
    k == 4  && return d == 0 ? zero(T) : T( d*cheb2(z[2]) )
    k == 5  && return d == 0 ? zero(T) : T( d*cheb3(z[2]) )

    # extension cost
    k == 6  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end





@inline function flowdθ(::Type{Val{:cheb3_dgt1}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,z[1],ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 4  && return d == 1 ? T( cheb0(z[2]) ) : zero(T)
    k == 5  && return d <= 1 ? zero(T) : T( d*cheb0(z[2]) )
    k == 6  && return d == 0 ? zero(T) : T( d*cheb1(z[2]) )
    k == 7  && return d == 0 ? zero(T) : T( d*cheb2(z[2]) )
    k == 8  && return d == 0 ? zero(T) : T( d*cheb3(z[2]) )

    # extension cost
    k == 9  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:cheb3_dgt1_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    # revenue
    k == 1  && return   d * rev_exp_restricted(θ[1], σ, z[1], ψ, Dgt0, geoid, roy)

    # drilling cost
    k == 2  && return d == 1 ? T( cheb0(z[2]) ) : zero(T)
    k == 3  && return d <= 1 ? zero(T) : T( d*cheb0(z[2]) )
    k == 4  && return d == 0 ? zero(T) : T( d*cheb1(z[2]) )
    k == 5  && return d == 0 ? zero(T) : T( d*cheb2(z[2]) )
    k == 6  && return d == 0 ? zero(T) : T( d*cheb3(z[2]) )

    # extension cost
    k == 7  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end




@inline function flowdθ(::Type{Val{:cheb3_cost}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    logp, logc, t = z

    # revenue
    k == 1  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy)
    k == 2  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy) * geoid
    k == 3  && return   d * rev_exp(1,θ[1],1,θ[2],θ[3],σ,logp,ψ,Dgt0,geoid,roy) * ( Dgt0 ? ψ : ψ*_ρ(σ) + θ[k]*(1-_ρ2(σ)))

    # drilling cost
    k == 4  && return d != 1 ? zero(T) : T(   cheb0(t) )
    k == 5  && return d <= 1 ? zero(T) : T( d*cheb0(t) )
    k == 6  && return d == 0 ? zero(T) : T( d*cheb1(t) )
    k == 7  && return d == 0 ? zero(T) : T( d*cheb2(t) )
    k == 8  && return d == 0 ? zero(T) : T( d*cheb3(t) )
    k == 9  && return d == 0 ? zero(T) : T( d*exp(logc) )

    # extension cost
    k == 10  && return d == 0 && sgn_ext ? one(T) : zero(T)

    throw(error("$k out of bounds"))
end



@inline function flowdθ(::Type{Val{:cheb3_cost_restr}}, θ::AbstractVector{T}, σ::T,     z::NTuple{N,T}, ψ::T, k::Integer,d::Integer,           d1::Integer, Dgt0::Bool, sgn_ext::Bool, geoid::Real, roy::T)::T where {N,T}
    d == 0 && !sgn_ext && return zero(T)

    logp, logc, t = z

    # revenue
    k == 1  && return   d * rev_exp_restricted(θ[1], σ, logp, ψ, Dgt0, geoid, roy)

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
