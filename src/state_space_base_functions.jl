# ------------------------------ new states --------------------------------

end_exp(dmx::Integer, Dmx::Integer, τmx::Integer)::Int = τmx+1
end_lrn(dmx::Integer, Dmx::Integer, τmx::Integer)::Int = τmx+dmx+2
end_inf(dmx::Integer, Dmx::Integer, τmx::Integer)::Int = τmx+dmx+2*Dmx+1

end_pts(d::Integer, D::Integer, t::Integer)::NTuple{4,Int} = (0, end_exp(d,D,t), end_lrn(d,D,t), end_inf(d,D,t))

ind_exp(ep::NTuple{4,Int}) = ep[2]   : -1 : ep[1]+1
ind_lrn(ep::NTuple{4,Int}) = ep[3]   : -1 : ep[2]+1
ind_inf(ep::NTuple{4,Int}) = ep[4]-1 : -1 : ep[3]+1

inf_fm_lrn(dmx::Int, ep::NTuple{4,Int})::StepRange{Int} = (ep[3]+1) .+ 2*(0:dmx-1)

exploratory_terminal(ep::NTuple{4,Int}) = ep[2]+1
exploratory_learning(ep::NTuple{4,Int}) = ep[2]+2 : ep[3]

dmx_exp(dmx::Integer, Dmx::Integer, τmx::Integer) = dmx
dmx_lrn(dmx::Integer, Dmx::Integer, τmx::Integer) = dmx_exp(dmx,Dmx,τmx)
dmx_inf(dmx::Integer, Dmx::Integer, τmx::Integer) = dmx

# --------------------- map observed state to an index --------------------

function state_idx(τ::Integer, D::Integer, d1::Integer, dmx::Integer, Dmx::Integer, τmx::Integer)::Int
    τ>=0           && return (τmx + 1) - τ
    D==0  && τ==-1 && return (τmx + 1) + (1 +  0 )
    D<Dmx && τ==-1 && return (τmx + 1) + (1 + dmx) + 1 + 2*(D - 1) + (1-d1)
    D==Dmx         && return (τmx + 1) + (1 + dmx) + 1 + 2*(Dmx-1)   # drop last +1 since we have d1=1 at terminal
    throw(error("invalid state"))
end

# --------------------- given a state index, return information --------------------

function _regime(s::Integer, endpts::NTuple{4,Int})::Symbol
  s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
  s <= endpts[2] && return :explore
  s <= endpts[3] && return :learn
  s <= endpts[4] && return :infill
  throw(DomainError())
end

function _D(s::Integer,endpts::NTuple{4,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[2] && return 0
    s <= endpts[3] && return s - endpts[2] - 1
    s <= endpts[4] && return (s - endpts[3] + isodd(s-endpts[3]) ) / 2
    throw(DomainError())
end

function _d1(s::Integer, endpts::NTuple{4,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[2] && return 0
    s <= endpts[3] && return s > endpts[2]+1
    s <= endpts[4] && return isodd(s-endpts[3])
end

function _τ(s::Integer, endpts::NTuple{4,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[2] && return endpts[2]-s
    return -1
end

function _horizon(s::Integer, endpts::NTuple{4,Int})::Symbol
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[2]    && return :Finite
    s == endpts[2]+1  && return :Terminal
    s <= endpts[3]    && return :Learning
    s <  endpts[4]    && return _sprime(s,0,endpts) == s ? :Infinite : :Finite
    s == endpts[4]    && return :Terminal
    throw(DomainError())
end

function _actionspace(s::Integer, dmx::Integer, Dmx::Integer, endpts::NTuple{4,Int})::UnitRange{Int}
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[2] && return 0:dmx
    s <= endpts[3] && return 0:0
    s <= endpts[4] && return 0:min((Dmx-_D(s,endpts)),dmx)
    throw(DomainError())
end

@inline _dp1space(s::Integer, dmx::Integer, Dmx::Integer, endpts::NTuple{4,Int}) = _actionspace(s,dmx,Dmx,endpts) .+ 1

function _first_action(s::Integer, dmx::Integer, Dmx::Integer, endpts::NTuple{4,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[2] && return 0
    s <= endpts[3] && return 0
    s <= endpts[4] && return 0
    throw(DomainError())
end


function _max_action(s::Integer, dmx::Integer, Dmx::Integer, endpts::NTuple{4,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[2] && return dmx
    s <= endpts[3] && return 0
    s <= endpts[4] && return min((Dmx-_D(s,endpts)),dmx)
    throw(DomainError())
end

function _num_actions(s::Integer, dmx::Integer, Dmx::Integer, endpts::NTuple{4,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[2] && return dmx+1
    s <= endpts[3] && return 1
    s <= endpts[4] && return min((Dmx-_D(s,endpts)),dmx)+1
    throw(DomainError())
end

@inline function _sprime(s::Integer, d::Integer, endpts::NTuple{4,Int})::Int
    if s <= endpts[1]
        s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    elseif s <= endpts[2]
        d == 0 && return s+1
        d >  0 && return endpts[2] + d + 1
    elseif s == endpts[2]+1
        return s
    elseif s <= endpts[3]
        return endpts[3] + 2*(s-endpts[2])-3  # was return endpts[3] + 2*d - 1. note d=(s-endpts[2]-1)
    elseif s == endpts[4]
        return s
    elseif s < endpts[4]
        d == 0  && return s + isodd(s-endpts[3])
        d >  0  && return s + 2*d - iseven(s-endpts[3])
    else
        throw(DomainError())
    end
end

# Generator expression
@inline _sprimes(s::Integer, dmx::Integer, Dmx::Integer, endpts::NTuple{4,Int}) = (_sprime(s,d,endpts) for d in _actionspace(s,dmx,Dmx,endpts))
