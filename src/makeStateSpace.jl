# __precompile__()
#
# "Construct state space, action spaces, and deterministic transitions"
# module makeStateSpace

# so we can add method for state type
import Base: ==, string, length, show

export
  well_problem,
  state,
  horizon, dmax, τmax, length, Dmax,
  sprime,
  state_idx,
  terminal_state_ind,
  explore_state_inds,
  infill_state_inds


# -------- Definition of the endogenous, deterministic state --------

"Deterministic state  (`τ`, `D`, `d1`)"
immutable state
  τ::Int  # "Time remaining in primary term"
  D::Int  # "Wells drilled to date"
  d1::Int # "Drilling last period == 1, no drilling last period == 0"
end

# Pretty print: https://docs.julialang.org/en/latest/manual/types.html#Custom-pretty-printing-1
string(s::state)::String = string((s.τ, s.D, s.d1))
show(io::IO, s::state) = print(io, string(s))

function ==(s1::state, s2::state)
  s1.τ == s2.τ    &&     s1.D == s2.D    &&    s1.d1 == s2.d1
end

# -------- Construct the state space --------

"Vector of endogenous deterministic states"
const state_space = Vector{state}

"Assemble state_space from max expiration `τmax` and max drilling `Dmax`"
function state_space(τmax::Int, Dmax::Int)::Vector{state}
  # Exploratory drilling
  a = [state( τ, 0, 0) for τ in τmax:-1:0]

  # Infill drilling with immediately prior drilling
  b = Dmax > 1    ?    [state(-1, D, d1) for D in 1:Dmax-1 for d1 in [1,0]]  :     Vector{state}(0)

  # Expiration / terminal
  c = [state(-1, Dmax, 1)]

  return state_space( [a..., b..., c..., ])
end


# -------- construct the action space --------

"Compute max # wells that can be drilled given state, dmax, and Dmax"
function max_action(s::state, dmax::Int, Dmax::Int)
  if (s.τ == -1) && (s.D ∈ [0, Dmax])
    return 0
  else
    return min(dmax, Dmax-s.D)
  end
end

"Vector of max # wells that can be drilled in ea. state"
function max_actions(SS::state_space, dmax::Int, Dmax::Int)
  [max_action(s, dmax, Dmax) for s in SS]
end

# -------- transition --------


"The state-transition rule"
function sprime(s::state, d::Int, Dmax::Int, dmax::Int)

  if d > dmax || d < 0
    throw(DomainError())
  end

  # NO DRILLING
  if d == 0

    # before expiration
    if s.τ > 0
      return state(s.τ-1, s.D, 0)

    # AT expiration. (Note, the d1 = 1 doesn't matter since is terminal)
    elseif s.τ == 0
      return state(-1, Dmax, 1)

    # In TERMINAL state w/ no drilling
    elseif s.τ < 0 && s.D == Dmax
      return state(-1, Dmax, 1)

    # AFTER expiration with no drilling
    else
      return state(-1, s.D, 0)
    end

  # ANY drilling
  elseif d > 0
    return state(-1, (s.D + d), 1)

  # Un-handled situations
  else
    error("Problem with state " * string(s))
  end
end


"Find position of state in state space"
function findstate(SS::state_space, s::state)
  for (i, ss) in enumerate(SS)
    if s == ss
      return i
    end
  end
  return 0
end

"Deterministic state transitions. out[i,j] = state next period if action j"
function state_transitions(SS::state_space, dmax::Integer, Dmax::Integer, τmax::Integer)

  out = zeros(Int, dmax + 1, length(SS))
  actions = [0:d for d in max_actions(SS, dmax, Dmax)]

  for (j, s) in enumerate(SS)
    for (i, d) in enumerate(actions[j])
      sp = sprime(s, d, Dmax, dmax)
      out[i,j] = state_idx(sp, Dmax, τmax)
    end
  end

  return out
end

# ----------- fast state transitions matrix ------------

"""
    horizon(Sprimes::Matrix, Γmax::Vector)

Return vector with horizon of problem at each state given matrix of feasible
state transitions `Sprimes` and the max feasible action `Γmax`

# Returns
  - `0` : Terminal/absorbing state (can ONLY remain in `SS[i]`; no need to compute value)
  - `-1`: Infinite (can remain in state `SS[i]`)
  - `1` : Finite (cannot remain in state `SS[i]`)
"""
function horizon(Sprimes::Matrix{Int}, Γmax::Vector{Int})
  n = length(Γmax)
  size(Sprimes,2) == n || throw(DimensionMismatch())
  out = Vector{Symbol}(n)
  for j in 1:n
    # no actions are possible
    if Γmax[j] == 0
      out[j] = :Terminal
    # staying in the state is still an option => infinite horizon
    elseif j == Sprimes[1, j]
      out[j] = :Infinite
    # otherwise, must be finite horizon
    else
      out[j] = :Finite
    end
  end
  return out
end




# -------- problem structure --------

"Structure of well problem"
immutable well_problem
  dmax::Int
  Dmax::Int
  τmax::Int
  SS::state_space
  Sprimes::Matrix{Int}
  Γmax::Vector{Int}
  horizon::Vector{Symbol}
end


"""
    well_problem(dmax, Dmax, τmax)

Construct the well problem given max per-period drilling `dmax`, max cumulative
drilling `Dmax` and primary term length `τmax`
"""
function well_problem(dmax::Int, Dmax::Int, τmax::Int)

  SS = state_space(τmax, Dmax)
  Sprimes = state_transitions(SS, dmax, Dmax, τmax)
  Γmax = max_actions(SS, dmax, Dmax)
  horz = horizon(Sprimes, Γmax)

  return well_problem(dmax, Dmax, τmax, SS, Sprimes, Γmax, horz)
end


# basic info about the problem
Base.length(p::well_problem) = length(p.SS)
dmax(       p::well_problem) = p.dmax
Dmax(       p::well_problem) = p.Dmax
τmax(       p::well_problem) = p.τmax

# methods for retrieving info from structure
horizon(p::well_problem, i::Int) = p.horizon[i]
dmax(   p::well_problem, i::Int) = p.Γmax[i]
state(  p::well_problem, i::Int) = p.SS[i]

"`action_iter(p::well_problem, i) = 0 : dmax(p, i)`"
action_iter( p::well_problem, i::Int) = 0 : dmax(p, i)

"Return vector of indices `iprime ∈ 1:length(SS)` such that `SS[iprime]` is reachable from `SS[i]`"
function sprime_idx(p::well_problem, i::Integer)
   return p.Sprimes[1:p.Γmax[i]+1, i]
end

action0(wp::well_problem,j::Integer) = wp.Sprimes[1,j]



"Find index of `s::state` given `wp::well_problem`"
function state_idx(s::state, Dmax::Integer, τmax::Integer)
  s.τ >= 0    && return τmax - s.τ + 1
  s.D == Dmax && return τmax+1 + (Dmax-1)*2 + 1
  if s.D < Dmax && s.τ == -1
    return τmax+1 + (s.D-1)*2 + (1-s.d1) + 1
  else
    throw(error("invalid state"))
  end
end


"Find index of `s::state` given `wp::well_problem`"
function state_idx(s::state, wp::well_problem)
  return state_idx(s, Dmax(wp), τmax(wp))
end

"Returns (dmax+1, iprimes, horizon) from `well_problem` in state `i`"
wp_info(wp::well_problem, i::Int) = (1:dmax(wp,i)+1, sprime_idx(wp,i), horizon(wp,i), wp.SS[i], )


function exploratory_dmax(wp::well_problem)
  1 ∈ explore_state_inds(wp)  || throw(error("Initial state is not exploratory."))
  return dmax(wp,1)
end

function infill_state_idx_from_exploratory(wp::well_problem)
  tmx = τmax(wp)
  dmx = dmax(wp,tmx)
  return tmx + 2*(1:dmx)
end

terminal_state_ind(wp::well_problem) = length(wp)
explore_state_inds(wp::well_problem) = τmax(wp)+1 : -1  : 1
infill_state_inds(wp::well_problem) =  τmax(wp)+1 + (Dmax(wp)*2-2 : -1 : 1)
