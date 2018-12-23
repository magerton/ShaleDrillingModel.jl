# ------------------------------ new states --------------------------------

# end_ex0(dmx::Integer, Dmx::Integer, τmx0::Integer)::Int = τmx0+1                            # τmx0+1
# end_lrn(dmx::Integer, Dmx::Integer, τmx0::Integer)::Int = end_exp(dmx,Dmx,τmx0) + (dmx+1)   # τmx0+dmx+2
# end_inf(dmx::Integer, Dmx::Integer, τmx0::Integer)::Int = end_lrn(dmx,Dmx,τmx0) + 2*Dmx-1   # τmx0+dmx+2*Dmx+1 = (τmx0+1) + (dmx+1) + 2*(Dmx-1)+1

# Functions to determine endpoints of each section in the state space
@inline end_ex1(d::Integer, D::Integer, t0::Integer, t1::Integer, e::Integer)::Int = (t1+1)
@inline end_ex0(d::Integer, D::Integer, t0::Integer, t1::Integer, e::Integer)::Int = end_ex1(d,D,t0,t1,e) + max(t0,e)+1
@inline end_lrn(d::Integer, D::Integer, t0::Integer, t1::Integer, e::Integer)::Int = end_ex0(d,D,t0,t1,e) + (d+1)
@inline end_inf(d::Integer, D::Integer, t0::Integer, t1::Integer, e::Integer)::Int = end_lrn(d,D,t0,t1,e) + 2*(D-1)+ 1
@inline strt_ex(d::Integer, D::Integer, t0::Integer, t1::Integer, e::Integer)::Int = end_ex0(d,D,t0,t1,e) - (e-1)     # t+d+2(D)+1 = (t+1) + (d+1) + 2*(D-1) + 1

# Assuming we have no lease extensions
@inline end_ex0(d::Integer, D::Integer, t0::Integer) = end_ex0(d,D,t0,-1,0)
@inline end_lrn(d::Integer, D::Integer, t0::Integer) = end_lrn(d,D,t0,-1,0)
@inline end_inf(d::Integer, D::Integer, t0::Integer) = end_inf(d,D,t0,-1,0)

# Create tuple of values that define state space
@inline end_pts(d,D,t0,t1,e)::NTuple{6,Int} = (0, end_ex1(d,D,t0,t1,e), end_ex0(d,D,t0,t1,e), end_lrn(d,D,t0,t1,e), end_inf(d,D,t0,t1,e), strt_ex(d,D,t0,t1,e),)
@inline end_pts(d,D,t0) = end_pts(d,D,t0,-1,0)

ind_ex1(ep::NTuple{6,Int}) = ep[2]   : -1 : ep[1]+1
ind_ex0(ep::NTuple{6,Int}) = ep[3]   : -1 : ep[2]+1
ind_exp(ep::NTuple{6,Int}) = ep[3]   : -1 : ep[1]+1
ind_lrn(ep::NTuple{6,Int}) = ep[4]   : -1 : ep[3]+1
ind_inf(ep::NTuple{6,Int}) = ep[5]-1 : -1 : ep[4]+1

inf_fm_lrn(dmx::Int, ep::NTuple{6,Int})::StepRange{Int} = (ep[4]+1) .+ 2*(0:dmx-1)

exploratory_terminal(ep::NTuple{6,Int}) = ep[3]+1
exploratory_learning(ep::NTuple{6,Int}) = ep[3]+2 : ep[4]

# What is this for???
# dmx_exp(dmx::Integer, Dmx::Integer, τmx::Integer) = dmx
# dmx_lrn(dmx::Integer, Dmx::Integer, τmx::Integer) = dmx_exp(dmx,Dmx,τmx)
# dmx_inf(dmx::Integer, Dmx::Integer, τmx::Integer) = dmx

max_ext(endpts::NTuple{6,Int}) = endpts[3]-endpts[6]+1

# --------------------- map observed state to an index --------------------

function state_idx(t1::Integer, t0::Integer, D::Integer, d1::Integer, dmx, Dmx, τ0mx, τ1mx, emx)::Int     # dmx::Integer, Dmx::Integer, t0mx::Integer, t1mx::Integer, e::Integer

    t1 >= 0 && t0 ∉ (0,emx) && throw(error("Cannot be in primary + extension simultaneously"))
    t1 >= 0         && return end_ex1(dmx, Dmx, τ0mx, τ1mx, emx) - t1
    t0 >= 0         && return end_ex0(dmx, Dmx, τ0mx, τ1mx, emx) - t0
    D==0  && t0==-1 && return end_ex0(dmx, Dmx, τ0mx, τ1mx, emx) + (1 + 0)
    D<Dmx && t0==-1 && return end_lrn(dmx, Dmx, τ0mx, τ1mx, emx) +  1 + 2*(D - 1) + (1-d1)
    D==Dmx          && return end_lrn(dmx, Dmx, τ0mx, τ1mx, emx) +  1 + 2*(Dmx-1)   # drop last +1 since we have d1=1 at terminal
    throw(error("invalid state"))
end

state_idx(t1::Integer, t0::Integer, D::Integer, d1::Integer, wp::well_problem) = state_idx(t1,t0,D,d1, wp.dmax, wp.Dmax, wp.τ0max, wp.τ1max, wp.ext)

# --------------------- given a state index, return information --------------------

function _regime(s::Integer, endpts::NTuple{6,Int})::Symbol
  s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
  s <= endpts[2] && return :primary_with_extension
  s <= endpts[3] && return :primary_or_extension
  s <= endpts[4] && return :learn
  s <= endpts[5] && return :infill
  throw(DomainError())
end

function _D(s::Integer,endpts::NTuple{6,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[3] && return 0
    s <= endpts[4] && return s - endpts[3] - 1
    s <= endpts[5] && return (s - endpts[4] + isodd(s-endpts[4]) ) / 2
    throw(DomainError())
end

function _d1(s::Integer, endpts::NTuple{6,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[3] && return 0
    s <= endpts[4] && return s > endpts[3]+1
    s <= endpts[5] && return isodd(s-endpts[4])
end

function _τ1(s::Integer, endpts::NTuple{6,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[2] && return endpts[2]-s
    return -1
end

function _τ0(s::Integer, endpts::NTuple{6,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[2] && return max_ext(endpts)
    s <= endpts[3] && return endpts[3]-s
    return -1
end

function _horizon(s::Integer, endpts::NTuple{6,Int})::Symbol
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[3]    && return :Finite
    s == endpts[3]+1  && return :Terminal
    s <= endpts[4]    && return :Learning
    s <  endpts[5]    && return _sprime(s,0,endpts) == s ? :Infinite : :Finite
    s == endpts[5]    && return :Terminal
    throw(DomainError())
end

function _actionspace(s::Integer, dmx::Integer, Dmx::Integer, endpts::NTuple{6,Int})::UnitRange{Int}
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[3] && return 0:dmx
    s <= endpts[4] && return 0:0
    s <= endpts[5] && return 0:min((Dmx-_D(s,endpts)),dmx)
    throw(DomainError())
end

@inline _actionspace(s::Integer, wp::well_problem) = _actionspace(s, wp.dmax, wp.Dmax, wp.endpts)


@inline _dp1space(s::Integer, dmx::Integer, Dmx::Integer, endpts::NTuple{6,Int}) = _actionspace(s,dmx,Dmx,endpts) .+ 1

function _first_action(s::Integer, dmx::Integer, Dmx::Integer, endpts::NTuple{6,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[3] && return 0
    s <= endpts[4] && return 0
    s <= endpts[5] && return 0
    throw(DomainError())
end

function _max_action(s::Integer, dmx::Integer, Dmx::Integer, endpts::NTuple{6,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[3] && return dmx
    s <= endpts[4] && return 0
    s <= endpts[5] && return min((Dmx-_D(s,endpts)),dmx)
    throw(DomainError())
end

@inline _max_action(s::Integer, wp::well_problem) = _max_action(s, wp.dmax, wp.Dmax, wp.endpts)


function _num_actions(s::Integer, dmx::Integer, Dmx::Integer, endpts::NTuple{6,Int})::Int
    s <= endpts[1] && throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    s <= endpts[3] && return dmx+1
    s <= endpts[4] && return 1
    s <= endpts[5] && return min((Dmx-_D(s,endpts)),dmx)+1
    throw(DomainError())
end

@inline function _sprime(s::Integer, d::Integer, endpts::NTuple{6,Int})::Int
    if s <= endpts[1]
        throw(DomainError(s, "s <= endpts[1] = $(endpts[1])"))
    elseif s == endpts[2]
        d == 0 && return endpts[6]
        d >  0 && return endpts[3] + d + 1
    elseif s <= endpts[3]
        d == 0 && return s+1
        d >  0 && return endpts[3] + d + 1
    elseif s == endpts[3]+1
        return s
    elseif s <= endpts[4]
        return endpts[4] + 2*(s-endpts[3])-3  # was return endpts[3] + 2*d - 1. note d=(s-endpts[2]-1)
    elseif s < endpts[5]
        d == 0  && return s + isodd(s-endpts[4])
        d >  0  && return s + 2*d - iseven(s-endpts[4])
    elseif s == endpts[5]
        return s
    else
        throw(DomainError())
    end
end

@inline _sprime(s::Integer, d::Integer, wp::well_problem) = _sprime(s,d,wp.endpts)

# Generator expression
@inline _sprimes(s::Integer, dmx::Integer, Dmx::Integer, endpts::NTuple{6,Int}) = (_sprime(s,d,endpts) for d in _actionspace(s,dmx,Dmx,endpts))
