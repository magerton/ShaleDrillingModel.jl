import Base: ==, string, length, show

export
  well_problem,
  state,
  horizon, dmax, τmax, length, Dmax, _nSexp,
  sprime,
  _sprimes,
  state_idx,
  terminal_state_ind,
  explore_state_inds,
  infill_state_inds

include("state_space_base_functions.jl")

# -------- new structs --------

"Deterministic state  (`τ1`, `τ0` `D`, `d1`)"
struct state
    τ1::Int  # Time remaining in initial term
    τ0::Int  # Time remaining in final term
    D::Int   # Wells drilled to date
    d1::Int  # Drilling last period == 1, no drilling last period == 0
end

"Structure of well problem"
struct well_problem
  dmax::Int
  Dmax::Int
  τ0max::Int
  τ1max::Int
  ext::Int
  endpts::NTuple{6,Int}
  SS::Vector{state}
  Sprime::Matrix{Int}
end

# ------------- methods for state ----------

# Pretty print: https://docs.julialang.org/en/latest/manual/types.html#Custom-pretty-printing-1
string(s::state)::String = string((s.τ1, s.τ0, s.D, s.d1))
show(io::IO, s::state) = print(io, string(s))

function ==(s1::state, s2::state)
    s1.τ1 == s2.τ1    &&     s1.τ0 == s2.τ0   &&     s1.D == s2.D    &&    s1.d1 == s2.d1
end

# ------------- methods for well_problem ----------

function state_space(dmax::Integer, Dmax::Integer, τ0max::Integer, τ1max::Integer, ext::Integer)::Vector{state}
    dmax < Dmax || throw(DomainError())
    # Primary term with extension
    exp1 = [state(τ1,ext,0,0) for τ1 in τ1max:-1:0]
    # Exploratory drilling (with terminal lease expiration)
    exp0 = [state(-1,τ0,0,0) for τ0 in τ0max:-1:-1]
    # Integrated (wrt information) infill
    inf_int = [state(-1,-1,D,1) for D in 1:dmax]  # τmax+1 + (0:dmax) (note overlap!)
    # Infill drilling with immediately prior drilling
    infill  = [state(-1,-1,D,d1) for D in 1:Dmax-1 for d1 in 1:-1:0] # τmax + 2 + dmax + (1:2*Dmax-2)
    # Infill terminal
    inf_term = [state(-1,-1,Dmax,1)] # τmax + 2 + dmax + 2*Dmax - 2 + 1  = τmax + dmax + 2*Dmax + 1
    return [exp1..., exp0..., inf_int..., infill..., inf_term...]
end

_num_actions(SS::Vector{state}, dmx::Integer, Dmx::Integer, endpts::NTuple{6,Int}) = [_num_actions(i, dmx, Dmx,endpts) for i in 1:length(SS)]

function well_problem(dmax::Integer, Dmax::Integer, τ0max::Integer, τ1max::Integer, ext::Integer)
    SS = state_space(dmax, Dmax, τ0max, τ1max, ext)
    ep = end_pts(dmax, Dmax, τ0max, τ1max, ext)
    Sprime = zeros(Int, dmx+1, ep[5])
    for j in 1:ep[5]
        for (di,d) in enumerate(_actionspace(j, dmx, Dmx, ep))
            Sprime[di,j] = _sprime(j,d,ep)
        end
    end
    return well_problem(dmx, Dmx, τ0mx, τ1mx, ext, ep , SS, Sprime)
end



# basic info about the problem
Base.length(p::well_problem) = p.endpts[5]
dmax(       p::well_problem) = p.dmax
Dmax(       p::well_problem) = p.Dmax
τmax(       p::well_problem) = p.τmax
endpts(     p::well_problem) = p.endpts

# retrieve info
horizon(p::well_problem, i::Integer) = _horizon(i,p.endpts)
state(  p::well_problem, i::Integer) = p.SS[i]
exploratory_dmax(p::well_problem) = dmx_exp(dmax(p), Dmax(p), τmax(p))

# size
_nS(    p::well_problem) = length(p)
_nSexp( p::well_problem) = p.endpts[4]
_nSint( p::well_problem) = length(ind_lrn(p.endpts))

# actions
action_iter(  p::well_problem, i::Integer) = _actionspace(p, i)
_num_actions( p::well_problem, i::Integer) = _num_actions( i, dmax(p), Dmax(p), p.endpts)
_actionspace( p::well_problem, i::Integer) = _actionspace( i, dmax(p), Dmax(p), p.endpts)
max_action(   p::well_problem, i::Integer) = _max_action(  i, dmax(p), Dmax(p), p.endpts)
_dp1space(    p::well_problem, i::Integer) = _dp1space(    i, dmax(p), Dmax(p), p.endpts)
_first_action(p::well_problem, i::Integer) = _first_action(i, dmax(p), Dmax(p), p.endpts)

_sprime(      p::well_problem, i::Integer, d::Integer) = _sprime(i, d, p.endpts)


action0(           p::well_problem, i::Integer) = _sprime( i, _first_action(p,i), p.endpts)
        sprime_itr(p::well_problem, i::Integer) = _sprimes(i, dmax(p), Dmax(p), p.endpts)
@inline sprime_idx(p::well_problem, i::Integer) = @views  p.Sprime[1:_num_actions(p,i), i]


exploratory_terminal(p::well_problem) = exploratory_terminal(p.endpts)
state_idx(p::well_problem, s::state) = state_idx(s.τ, s.D, s.d1, dmax(p), Dmax(p), τmax(p))
wp_info(p::well_problem, i::Integer) = (_dp1space(p,i), sprime_idx(p,i), horizon(p,i), state(p,i), )


terminal_state_ind(p::well_problem) = p.endpts[5]


inf_fm_lrn(        p::well_problem) = inf_fm_lrn(dmax(p), p.endpts)
ind_inf(           p::well_problem) = ind_inf(p.endpts)
infill_state_inds( p::well_problem) = ind_inf(p.endpts)
explore_state_inds(p::well_problem) = ind_exp(p.endpts)
learn_state_inds(  p::well_problem) = ind_lrn(p.endpts)
exploratory_learning(p::well_problem) = exploratory_learning(p.endpts)


# -------- transition --------

"The state-transition rule"
function OLDsprime(s::state, d::Int, Dmax::Int, dmax::Int)
    0 <= d <= dmax <= Dmax ||  println("bad $d, $s combo with $dmax, $Dmax")
    if d == 0                                                      # NO DRILLING
        s.τ >= 0 &&                  return state(s.τ-1, s.D, 0)   # before expiration
        s.τ <  0 && s.D == Dmax  &&  return state(-1, Dmax, 1)     # In TERMINAL state w/ no drilling
        return state(-1, s.D, 0)                                   # AFTER expiration with no drilling
    elseif d > 0                                                   # ANY drilling
        return state(-1, (s.D + d), 1)
    else                                                           # Un-handled situations
        error("Problem with state " * string(s))
    end
end
