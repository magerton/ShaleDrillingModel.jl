import Base: ==, string, length, show

export AbstractUnitProblem,
    LeasedProblem,
    LeasedProblemContsDrill,
    PerpetualProblem,
    sprime,
    ssprime,
    state,
    state_idx,
    end_ex1,
    end_ex0,
    end_lrn,
    end_inf

abstract type AbstractUnitProblem end

struct LeasedProblem <: AbstractUnitProblem
    dmax::Int
    Dmax::Int
    τ0max::Int
    τ1max::Int
    ext::Int
end

struct LeasedProblemContsDrill <: AbstractUnitProblem
    dmax::Int
    Dmax::Int
    τ0max::Int
    τ1max::Int
    ext::Int
end

struct PerpetualProblem <: AbstractUnitProblem
    dmax::Int
    Dmax::Int
end

PerpetualProblem(dmax,Dmax,τ0max,τ1max,ext) = PerpetualProblem(dmax,Dmax)

_nstates_per_D(wp::AbstractUnitProblem) = 1
_nstates_per_D(wp::LeasedProblemContsDrill) = 2

_dmax(  wp::AbstractUnitProblem) = wp.dmax
_Dmax(  wp::AbstractUnitProblem) = wp.Dmax

_τ0max( wp::AbstractUnitProblem) = wp.τ0max
_τ1max( wp::AbstractUnitProblem) = wp.τ1max
_ext(   wp::AbstractUnitProblem) = wp.ext

_τ0max( wp::PerpetualProblem) = -1
_τ1max( wp::PerpetualProblem) = -1
_ext(   wp::PerpetualProblem) = 0

# ----------------------------------
# endpoints of each set of states
#   end_ex1 - Expiration of first lease term
#   end_ex0 - End of last lease term (expiration if present, or regular lease if not)
#   end_lrn - In learning stage, we have drilled _dmax(wp) wells
#   end_inf - Terminal state -- we've drilled everything
#   strt_ex - State where extension starts
# ----------------------------------
@inline end_ex1(wp::LeasedProblemContsDrill) = _τ1max(wp)+1
@inline end_ex0(wp::LeasedProblemContsDrill) = end_ex1(wp) + max(_τ0max(wp),_ext(wp))+1
@inline end_lrn(wp::LeasedProblemContsDrill) = end_ex0(wp) + _dmax(wp)+1
@inline end_inf(wp::LeasedProblemContsDrill) = end_lrn(wp) + 2*(_Dmax(wp)-1)+ 1
@inline strt_ex(wp::LeasedProblemContsDrill) = end_ex0(wp) - (_ext(wp)-1)     # t+d+2(D)+1 = (t+1) + (d+1) + 2*(D-1) + 1

@inline end_ex1(wp::LeasedProblem) = _τ1max(wp)+1
@inline end_ex0(wp::LeasedProblem) = end_ex1(wp) + max(_τ0max(wp),_ext(wp))+1
@inline end_lrn(wp::LeasedProblem) = end_ex0(wp) + _dmax(wp)+1
@inline end_inf(wp::LeasedProblem) = end_lrn(wp) + _Dmax(wp)
@inline strt_ex(wp::LeasedProblem) = end_ex0(wp) - (_ext(wp)-1)     # t+d+2(D)+1 = (t+1) + (d+1) + 2*(D-1) + 1

@inline end_ex1(wp::PerpetualProblem) = 0
@inline end_ex0(wp::PerpetualProblem) = 1
@inline end_lrn(wp::PerpetualProblem) = end_ex0(wp) + _dmax(wp)
@inline end_inf(wp::PerpetualProblem) = end_lrn(wp) + _Dmax(wp)
@inline strt_ex(wp::PerpetualProblem) = 1

# ----------------------------------
# length of states
# ----------------------------------

Base.length(wp::AbstractUnitProblem) = end_inf(wp)
_nS(    wp::AbstractUnitProblem) = length(wp)
_nSexp( wp::AbstractUnitProblem) = end_lrn(wp)

# ----------------------------------
# regions of state space where we have learning
# ----------------------------------

exploratory_terminal(wp::AbstractUnitProblem) = end_ex0(wp)+1
exploratory_learning(wp::AbstractUnitProblem) = end_ex0(wp)+2 : end_lrn(wp)

exploratory_terminal(wp::PerpetualProblem) = end_ex0(wp)
exploratory_learning(wp::PerpetualProblem) = end_ex0(wp)+1 : end_lrn(wp)

# ----------------------------------
# indices of where we are in things
# ----------------------------------

@inline ind_ex1(wp::AbstractUnitProblem) = end_ex1(wp)   : -1 : 1
@inline ind_ex0(wp::AbstractUnitProblem) = end_ex0(wp)   : -1 : end_ex1(wp)+1
@inline ind_exp(wp::AbstractUnitProblem) = end_ex0(wp)   : -1 : 1
@inline ind_lrn(wp::AbstractUnitProblem) = end_lrn(wp)   : -1 : end_ex0(wp)+1
@inline ind_inf(wp::AbstractUnitProblem) = end_inf(wp)-1 : -1 : end_lrn(wp)+1

@inline inf_fm_lrn(wp::AbstractUnitProblem) = (end_lrn(wp)+1) .+ _nstates_per_D(wp)*(0:_dmax(wp)-1)

# ----------------------------------
# which state are we in?
# ----------------------------------

function state_idx(wp::LeasedProblemContsDrill, t1::Integer, t0::Integer, D::Integer, d1::Integer)::Int
    t1 >= 0 && t0 ∉ (0,_ext(wp)) && throw(error("Cannot be in primary + extension simultaneously"))
    t1 >= 0               && return end_ex1(wp) - t1
    t0 >= 0               && return end_ex0(wp) - t0
    D==0  && t0==-1       && return end_ex0(wp) + (1 + 0)
    D<_Dmax(wp) && t0==-1 && return end_lrn(wp) +  1 + 2*(D - 1) + (1-d1)
    D==_Dmax(wp)          && return end_lrn(wp) +  1 + 2*(_Dmax(wp)-1)   # drop last +1 since we have d1=1 at terminal
    throw(error("invalid state"))
end

function state_idx(wp::LeasedProblem, t1::Integer, t0::Integer, D::Integer, d1::Integer)::Int
    t1 >= 0 && t0 ∉ (0,_ext(wp)) && throw(error("Cannot be in primary + extension simultaneously"))
    t1 >= 0                && return end_ex1(wp) - t1
    t0 >= 0                && return end_ex0(wp) - t0
    D==0   && t0==-1       && return end_ex0(wp) + (1 + 0)
    D<=_Dmax(wp) && t0==-1 && return end_lrn(wp) + D
    throw(error("invalid state"))
end

function state_idx(wp::PerpetualProblem, t1::Integer, t0::Integer, D::Integer, d1::Integer)::Int
    D==0         && return end_ex0(wp)
    D<=_Dmax(wp) && return end_lrn(wp) + D
    throw(error("invalid state"))
end

# ----------------------------------
# information about state
# ----------------------------------

function _regime(wp::AbstractUnitProblem, s::Integer)::Symbol
    s <= 0 && throw(DomainError(s, "s <= 0"))
    s <= end_ex1(wp) && return :primary_WITH_extension
    s <= end_ex0(wp) && return :primary_or_extension
    s == end_ex0(wp) + 1 && return :expired
    s <= end_lrn(wp) && return :learn
    s <= end_inf(wp) && return :infill
    throw(DomainError())
end

function _horizon(wp::PerpetualProblem, sidx::Integer)::Symbol
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp)  && return :Infinite
    sidx <= end_lrn(wp)  && return :Learning
    sidx <  end_inf(wp)  && return :Infinite
    sidx == end_inf(wp)  && return :Terminal
    throw(DomainError())
end

function _horizon(wp::LeasedProblem, sidx::Integer)::Symbol
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp)   && return :Finite
    sidx == end_ex0(wp)+1 && return :Terminal
    sidx <= end_lrn(wp)   && return :Learning
    sidx <  end_inf(wp)   && return :Infinite
    sidx == end_inf(wp)   && return :Terminal
    throw(DomainError())
end

function _horizon(wp::LeasedProblemContsDrill, sidx::Integer)::Symbol
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp)   && return :Finite
    sidx == end_ex0(wp)+1 && return :Terminal
    sidx <= end_lrn(wp)   && return :Learning
    sidx <  end_inf(wp)   && return isodd(end_inf(wp)-sidx) ? :Infinite : :Finite
    sidx == end_inf(wp)   && return :Terminal
    throw(DomainError())
end


function _D(wp::LeasedProblemContsDrill, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp) && return 0
    sidx <= end_lrn(wp) && return  sidx - end_ex0(wp) - 1
    sidx <= end_inf(wp) && return (sidx - end_lrn(wp) + isodd(sidx-end_lrn(wp)) ) / 2
    throw(DomainError())
end

function _D(wp::LeasedProblem, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp) && return 0
    sidx <= end_lrn(wp) && return sidx - end_ex0(wp) - 1
    sidx <= end_inf(wp) && return sidx - end_lrn(wp)
    throw(DomainError())
end

function _D(wp::PerpetualProblem, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp) && return 0
    sidx <= end_lrn(wp) && return sidx - end_ex0(wp)
    sidx <= end_inf(wp) && return sidx - end_lrn(wp)
    throw(DomainError())
end


_d1(wp::AbstractUnitProblem, sidx::Integer) = 0

function _d1(wp::LeasedProblemContsDrill, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp)+1 && return 0
    sidx <= end_lrn(wp)   && return 1
    sidx <  end_inf(wp)   && return isodd(sidx-end_lrn(wp))
    sidx == end_inf(wp)   && return 1
end


function _τ1(wp::AbstractUnitProblem, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex1(wp) && return end_ex1(wp)-sidx
    return -1
end

function _τ0(wp::AbstractUnitProblem, sidx::Integer)::Int
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex1(wp) && return _ext(wp)
    sidx <= end_ex0(wp) && return end_ex0(wp)-sidx
    return -1
end

_τ1(wp::PerpetualProblem, sidx::Integer) = -1
_τ0(wp::PerpetualProblem, sidx::Integer) = -1


"Deterministic state  (`τ1`, `τ0` `D`, `d1`)"
struct state
    τ1::Int  # Time remaining in initial term
    τ0::Int  # Time remaining in final term
    D::Int   # Wells drilled to date
    d1::Int  # Drilling last period == 1, no drilling last period == 0
end

# ------------- methods for state ----------

# Pretty print: https://docs.julialang.org/en/latest/manual/types.html#Custom-pretty-printing-1
string(s::state)::String = string((s.τ1, s.τ0, s.D, s.d1))
show(io::IO, s::state) = print(io, string(s))

function ==(s1::state, s2::state)
    s1.τ1 == s2.τ1    &&     s1.τ0 == s2.τ0   &&     s1.D == s2.D    &&    s1.d1 == s2.d1
end

state(wp::AbstractUnitProblem, s::Integer) = state(_τ1(wp,s), _τ0(wp,s), _D(wp,s), _d1(wp,s))

_d1(s::state) = s.d1
_τ1(s::state) = s.τ1
_τ0(s::state) = s.τ0
_D(s::state) = s.D
_sgnext(s::state) = s.τ1 == 0 && s.τ0 > 0
_τrem(s::state) = max(s.τ1,0) + max(s.τ0,0)
_τ11(s::state, wp::AbstractUnitProblem) = 2*_τrem(s)/maxlease(wp)-1
_Dgt0(wp::AbstractUnitProblem, sidx::Integer) = sidx > end_ex0(wp)+1
_Dgt0(s::state) = s.D > 0

stateinfo(wp::AbstractUnitProblem, st::state) = (_d1(st), _Dgt0(st), _sgnext(st), _τ11(st, wp),)

_sgnext(wp::AbstractUnitProblem, sidx::Integer) = sidx == end_ex1(wp)
function _sign_lease_extension(sidx::Integer,wp::AbstractUnitProblem)
    @warn "Deprecated. use `sgnext(wp,sidx)`"
    return _sgnext(wp,sidx)
end


function _τrem(wp::AbstractUnitProblem, sidx::Integer)
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex1(wp) && return end_ex1(wp)-sidx + _ext(wp)+1
    sidx <= end_ex0(wp) && return end_ex0(wp)-sidx
    return -1
end

_τrem(wp::PerpetualProblem,sidx::Integer) = -1

max_ext(wp::AbstractUnitProblem) = end_ex0(wp)-strt_ex(wp)+1

maxlease(wp::AbstractUnitProblem) = max( end_ex1(wp)+max_ext(wp), end_ex0(wp)-end_ex1(wp) ) - 1


_τ11(wp::AbstractUnitProblem, sidx::Integer) = 2*_τrem(wp,sidx)/maxlease(wp)-1


# ----------------------------------
# actions
# ----------------------------------

function _dmax(wp::AbstractUnitProblem, sidx::Integer)
    sidx <= 0 && throw(DomainError(sidx, "s <= 0"))
    sidx <= end_ex0(wp) && return _dmax(wp)
    sidx <= end_lrn(wp) && return 0
    sidx <  end_inf(wp) && return min((_Dmax(wp)-_D(wp,sidx)),_dmax(wp))
    sidx == end_inf(wp) && return 0
    throw(DomainError())
end

@inline actionspace(wp::AbstractUnitProblem, sidx::Integer) = 0:_dmax(wp,sidx)
@inline dp1space(   wp::AbstractUnitProblem, sidx::Integer) = actionspace(wp,sidx) .+ 1


# ----------------------------------
# states
# ----------------------------------

function state_space_vector(wp::LeasedProblemContsDrill)::Vector{state}
    _dmax(wp) <= _Dmax(wp) || throw(DomainError("dmax=$_dmax(wp), Dmax=$_Dmax(wp)"))
    # Primary term with extension
    exp1 = [state(τ1,_ext(wp),0,0) for τ1 in _τ1max(wp):-1:0]
    # Exploratory drilling (with terminal lease expiration)
    exp0 = [state(-1,τ0,0,0) for τ0 in _τ0max(wp):-1:-1]
    # Integrated (wrt information) infill
    inf_int = [state(-1,-1,D,1) for D in 1:_dmax(wp)]  # τmax+1 + (0:dmax) (note overlap!)
    # Infill drilling with immediately prior drilling
    infill  = [state(-1,-1,D,d1) for D in 1:_Dmax(wp)-1 for d1 in 1:-1:0] # τmax + 2 + dmax + (1:2*Dmax-2)
    # Infill terminal
    inf_term = [state(-1,-1,_Dmax(wp),1)] # τmax + 2 + dmax + 2*Dmax - 2 + 1  = τmax + dmax + 2*Dmax + 1
    return [exp1..., exp0..., inf_int..., infill..., inf_term...]
end

function state_space_vector(wp::LeasedProblem)::Vector{state}
    _dmax(wp) <= _Dmax(wp) || throw(DomainError("dmax=$_dmax(wp), Dmax=$_Dmax(wp)"))
    # Primary term with extension
    exp1 = [state(τ1,_ext(wp),0,0) for τ1 in _τ1max(wp):-1:0]
    # Exploratory drilling (with terminal lease expiration)
    exp0 = [state(-1,τ0,0,0) for τ0 in _τ0max(wp):-1:-1]
    # Integrated (wrt information) infill
    inf_int = [state(-1,-1,D,0) for D in 1:_dmax(wp)]  # τmax+1 + (0:dmax) (note overlap!)
    # Infill drilling with immediately prior drilling
    infill  = [state(-1,-1,D,0) for D in 1:_Dmax(wp)] # τmax + 2 + dmax + (1:2*Dmax-2)
    return [exp1..., exp0..., inf_int..., infill...]
end

function state_space_vector(wp::PerpetualProblem)::Vector{state}
    _dmax(wp) <= _Dmax(wp) || throw(DomainError("dmax=$_dmax(wp), Dmax=$_Dmax(wp)"))
    # Primary term with extension
    exp1 = [state(τ1,_ext(wp),0,0) for τ1 in _τ1max(wp):-1:0]
    # Exploratory drilling (with terminal lease expiration)
    exp0 = [state(-1,τ0,0,0) for τ0 in _τ0max(wp):-1:-1]
    # Integrated (wrt information) infill
    inf_int = [state(-1,-1,D,0) for D in 1:_dmax(wp)]
    # Infill drilling with immediately prior drilling
    infill  = [state(-1,-1,D,0) for D in 1:_Dmax(wp)]
    return [exp1..., exp0..., inf_int..., infill...]
end

# ----------------------------------
# sprime
# ----------------------------------

@inline function sprime(wp::LeasedProblemContsDrill, s::Integer, d::Integer)::Int
    if s <= 0
        throw(DomainError(s, "s <= 0"))
    elseif s == end_ex1(wp)
        d == 0 && return strt_ex(wp)
        d >  0 && return end_ex0(wp) + d + 1
    elseif s <= end_ex0(wp)
        d == 0 && return s+1
        d >  0 && return end_ex0(wp) + d + 1
    elseif s == end_ex0(wp)+1
        return s
    elseif s <= end_lrn(wp)
        return end_lrn(wp) + 2*(s-end_ex0(wp))-3  # was return endpts[3] + 2*d - 1. note d=(s-endpts[2]-1)
    elseif s < end_inf(wp)
        d == 0  && return s + isodd(s-end_lrn(wp))
        d >  0  && return s + 2*d - iseven(s-end_lrn(wp))
    elseif s == end_inf(wp)
        return s
    else
        throw(DomainError())
    end
end

@inline function sprime(wp::LeasedProblem, s::Integer, d::Integer)::Int
    if s <= 0
        throw(DomainError(s, "s <= 0"))
    elseif s == end_ex1(wp)
        d == 0 && return strt_ex(wp)
        d >  0 && return end_ex0(wp) + d + 1
    elseif s <= end_ex0(wp)
        d == 0 && return s+1
        d >  0 && return end_ex0(wp) + d + 1
    elseif s == end_ex0(wp)+1
        return s
    elseif s <= end_lrn(wp)
        return end_lrn(wp) + (s-end_ex0(wp)-1)
    elseif s < end_inf(wp)
        return s + d
    elseif s == end_inf(wp)
        return s
    else
        throw(DomainError())
    end
end

@inline function sprime(wp::PerpetualProblem, s::Integer, d::Integer)::Int
    if s <= 0
        throw(DomainError(s, "s <= 0"))
    elseif s == end_ex0(wp)
        d == 0 && return s
        d >  0 && return end_ex0(wp) + d
    elseif s <= end_lrn(wp)
        return end_lrn(wp) + (s-end_ex0(wp))
    elseif s < end_inf(wp)
        return s + d
    elseif s == end_inf(wp)
        return s
    else
        throw(DomainError())
    end
end

@inline sprimes(wp::AbstractUnitProblem, sidx::Integer) = (sprime(wp,sidx,d) for d in actionspace(wp,sidx))

"retrieve next state, skipping through learning if necessary."
function ssprime(wp::AbstractUnitProblem, s::Integer, d::Integer)::Int
    if s <= 0
        throw(DomainError(s, "s <= 0"))
    elseif s <= end_ex0(wp)
        return d == 0 ? sprime(wp,s,d) : sprime(wp, sprime(wp,s,d), d)  # because of LEARNING transition
    elseif s <= end_inf(wp)
        return sprime(wp,s,d)
    else
        throw(DomainError(s, "s > end_inf(wp)"))
    end
end


# ----------------------------------
# other functions
# ----------------------------------

# wp_info(wp::AbstractUnitProblem, s::Integer) = (dp1space(wp,s), sprimes(wp,s), _horizon(wp,s), state(wp,s), )
# action0(wp::AbstractUnitProblem,s::Integer) = sprime(wp, s, 0)
# stateinfo(wp::AbstractUnitProblem, s::Integer) = (_d1(wp,s), _Dgt0(wp,s), _sgnext(wp,s), _τ11(wp,s),)
# stateinfo(s::state) = (_d1(s), _Dgt0(s), _sgnext(s), _τ11(s),)

# infill_state_inds( wp::AbstractUnitProblem) = ind_inf(wp)
# explore_state_inds(wp::AbstractUnitProblem) = ind_exp(wp)


# end_pts(wp::AbstractUnitProblem) = (0, end_ex1(wp), end_ex0(wp), end_lrn(wp), end_inf(wp), strt_ex(wp),)
# _nSint( wp::AbstractUnitProblem) = end_lrn(wp) - end_ex0(wp)





# export
# ------------
# _dmax,
# _Dmax,
# _τ0max,
# _τ1max,
# _ext,
# strt_ex,
# _nS,
# _nSexp,
# ind_ex1,
# ind_ex0,
# ind_exp,
# ind_lrn,
# ind_inf,
# inf_fm_lrn,
# _horizon,
# _D,
# _d1,
# _τ1,
# _τ0,
# _Dgt0,
# _sgnext,
# _sign_lease_extension,
# _τrem,
# maxlease,
# _τ11,
# _dmax,
# actionspace,
# dp1space,
# state_space_vector,
# wp_info,
# action0
