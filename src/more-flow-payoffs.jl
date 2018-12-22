

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
