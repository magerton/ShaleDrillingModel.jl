export ItpSharedEV, logP!

struct ItpSharedEV{A1<:Interpolations.AbstractInterpolation,A2<:Interpolations.AbstractInterpolation,A3<:Interpolations.AbstractInterpolation,TT<:Tuple}
  EV::A1
  dEV::A2
  dEVσ::A3
  itypes::TT
end

function ItpSharedEV(sev::SharedEV{T,N,N2}, p::dcdp_primitives, σ::Real) where {T,N,N2}

    ntyp = length(sev.itypes)
    nθ = size(sev.dEV, N2-ntyp-1)
    nv = _nv(p)
    nS = _nS(p)
    nSexp1 = _nSexp(p)+1

    scalegrid(x::Range{T}) where {T<:AbstractFloat} = x
    scalegrid(x::Integer) = 1:x
    scalegrid(x::AbstractVector) = 1:length(x)
    splinetype(r::Union{StepRange,StepRangeLen}, flag::Interpolations.Flag=Linear()) = BSpline(flag)
    splinetype(r::UnitRange                    , flag::Interpolations.Flag=Linear()) = NoInterp()

    scl_EV   = scalegrid.((p.zspace..., _ψspace(p,σ),             nS,     sev.itypes...))
    scl_dEV  = scalegrid.((p.zspace..., _ψspace(p,σ), nθ,         nS,     sev.itypes...))
    scl_dEVσ = scalegrid.((p.zspace..., _ψspace(p,σ), _vspace(p), nSexp1, sev.itypes...))

    it_EV    = interpolate!(sev.EV  , splinetype.(scl_EV  ), OnGrid())
    it_dEV   = interpolate!(sev.dEV , splinetype.(scl_dEV ), OnGrid())
    it_dEVσ  = interpolate!(sev.dEVσ, splinetype.(scl_dEVσ), OnGrid())

    sit_EV   = Interpolations.scale(it_EV    , scl_EV...)
    sit_dEV  = Interpolations.scale(it_dEV   , scl_dEV...)
    sit_dEVσ = Interpolations.scale(it_dEVσ  , scl_dEVσ...)

    return ItpSharedEV(sit_EV, sit_dEV, sit_dEVσ, sev.itypes)
end

function ItpSharedEV(σ::Real)
  global g_SharedEV
  global g_dcdp_primitives
  return ItpSharedEV(g_SharedEV, g_dcdp_primitives, σ)
end

@GenGlobal g_ItpSharedEV

# TODO: MAKE SURE to check if data is within the indices

unsafegetityp(grid::Range, idx::Real) = idx
unsafegetityp(grid::Vector, idx::Integer) = grid[idx]

getitype(grid::Vector, idx::Integer) = unsafegetityp(grid, idx)
function getitype(grid::Range, idx::Real)
    minimum(grid) <= idx <= maximum(grid) || throw(DomainError())
    unsafegetityp(grid, idx)
end


function unsafegetityp(sev::Union{SharedEV,ItpSharedEV}, itypidx::Tuple)
    length(sev.itypes) == length(itypidx) || throw(DimensionMismatch())
    return unsafegetityp.(sev.itypes, itypidx)
end

function logP!(grad::AbstractVector{T}, tmp::AbstractVector, θfull::AbstractVector, prim::dcdp_primitives, isev::ItpSharedEV, dograd::Bool, itypidx::Tuple, uv::NTuple{2,<:Real}, dp1::Integer, s_idx::Integer, z::Real...) where {T}

  # unpack information about current state
  roy, geo = getitype.(isev.itypes, itypidx)
  omroy = one(T) - roy

  # gradient & coef views
  θt =    @view(θfull[[geo, prim.ngeo+1:end-1...]])
  σ = θfull[end]

  # states we can iterate over
  s = state(prim, s_idx)
  Dgt0 = s.D > 0

  # information
  ψ = Dgt0 ? uv[1] : uv[1] + σ*uv[2] # , extrema(_ψspace(prim, σ))...)
  v = uv[2]

  # containers
  dmxp1rng = Base.OneTo(dmax(prim, s_idx)+1)
  ubV = @view(tmp[dmxp1rng])

  # @inbounds
  for di in dmxp1rng
    s_idxp = prim.wp.Sprimes[di,s_idx]
    ubV[di] = 0.0*prim.f(θt, σ, z..., ψ, di-1, s.d1, Dgt0, omroy) + prim.β * isev.EV[z..., ψ, s_idxp, itypidx...]
  end

  dograd ||  return ubV[dp1] # - logsumexp(ubV)

  logp = ubV[dp1] # - logsumexp_and_softmax!(ubV)
  nSexp1 = _nSexp(prim)+1
  gradvw = @view(grad[[geo, prim.ngeo+1:end...]])

  # TODO: create a proper function mapping sprime_idx to sprime_idx_for_σ given s_idx & wp/prim

  # @inbounds
  for di in dmxp1rng
    wt = di==dp1 ? one(T) : zero(T) # -ubV[di] : -ubV[di]
    dim1 = di-1
    s_idxp = prim.wp.Sprimes[di,s_idx]
    for k in Base.OneTo(length(θt))
      # gk = k == 1 ? geo : prim.ngeo + k - 1
      gradvw[k]        += wt * (0.0*prim.df( θt, σ, z..., ψ, k, dim1,   s.d1, Dgt0, omroy) + prim.β * isev.dEV[ z..., ψ, k, s_idxp, itypidx...] )
    end
    Dgt0 || (grad[end] += wt * (0.0*prim.dfσ(θt, σ, z..., ψ, v, dim1,               omroy) + prim.β * isev.dEVσ[z..., ψ, v, min(s_idxp,nSexp1), itypidx...] ))
  end

  return logp
end
