export ItpSharedEV

struct ItpSharedEV{A1<:Interpolations.AbstractInterpolation,A2<:Interpolations.AbstractInterpolation,A3<:Interpolations.AbstractInterpolation,TT<:Tuple}
  EV::A1
  dEV::A2
  dEVσ::A3
  dEVψ::A3
  itypes::TT
end

function ItpSharedEV(sev::SharedEV{T,N,N2}, p::dcdp_primitives, σ::Real, flag::Interpolations.Flag=Linear()) where {T,N,N2}

    ntyp = length(sev.itypes)
    nθ = size(sev.dEV, N2-ntyp-1)
    nS = _nS(p)
    nSexp1 = _nSexp(p)+1

    scalegrid(x::Range{T}) where {T<:AbstractFloat} = x
    scalegrid(x::Integer) = 1:x
    scalegrid(x::AbstractVector) = 1:length(x)
    splinetype(r::Union{StepRange,StepRangeLen}, flag::Interpolations.Flag) = BSpline(flag)
    splinetype(r::UnitRange                    , flag::Interpolations.Flag) = NoInterp()

    scl_EV   = scalegrid.((p.zspace..., _ψspace(p,σ),         nS, sev.itypes...))
    scl_dEV  = scalegrid.((p.zspace..., _ψspace(p,σ), nθ,     nS, sev.itypes...))
    scl_dEVσ = scalegrid.((p.zspace..., _ψspace(p,σ),     nSexp1, sev.itypes...))

    it_EV    = interpolate!(sev.EV  , splinetype.(scl_EV  , flag), OnGrid())
    it_dEV   = interpolate!(sev.dEV , splinetype.(scl_dEV , flag), OnGrid())
    it_dEVσ  = interpolate!(sev.dEVσ, splinetype.(scl_dEVσ, flag), OnGrid())
    it_dEVψ  = interpolate!(sev.dEVψ, splinetype.(scl_dEVσ, flag), OnGrid())

    sit_EV   = Interpolations.scale(it_EV    , scl_EV...)
    sit_dEV  = Interpolations.scale(it_dEV   , scl_dEV...)
    sit_dEVσ = Interpolations.scale(it_dEVσ  , scl_dEVσ...)
    sit_dEVψ = Interpolations.scale(it_dEVψ  , scl_dEVσ...)

    return ItpSharedEV(sit_EV, sit_dEV, sit_dEVσ, sit_dEVψ, sev.itypes)
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
