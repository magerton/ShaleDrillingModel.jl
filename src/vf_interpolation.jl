export ItpSharedEV

struct ItpSharedEV{T,A1<:Interpolations.AbstractInterpolation{T},A2<:Interpolations.AbstractInterpolation{T},A3<:Interpolations.AbstractInterpolation{T},TT<:Tuple}
  EV::A1
  dEV::A2
  dEVσ::A3
  dEVψ::A3
  itypes::TT
end

function ItpSharedEV(sev::SharedEV{T,N,N2,TT}, p::dcdp_primitives, σ::Real=1.0, flag::Interpolations.Flag=Linear()) where {T,N,N2,TT}

    ntyp = length(sev.itypes)
    nθ = _nθt(p)
    nS = _nS(p)
    nSexp1 = _nSexp(p)

    scalegrid(x::Range{S}) where {S<:AbstractFloat} = x
    scalegrid(x::Integer) = Base.OneTo(x)
    scalegrid(x::AbstractVector) = Base.OneTo(length(x))

    splinetype(r::Union{StepRange,StepRangeLen}, flag::Interpolations.Flag) = BSpline(flag)
    splinetype(r::AbstractUnitRange            , flag::Interpolations.Flag) = NoInterp()

    scl_EV   = scalegrid.((p.zspace..., _ψspace(p,σ),         nS, sev.itypes...))
    scl_dEV  = scalegrid.((p.zspace..., _ψspace(p,σ), nθ,     nS, sev.itypes...))
    scl_dEVσ = scalegrid.((p.zspace..., _ψspace(p,σ),     nSexp1, sev.itypes...))

    it_EV    = interpolate!(sev.EV  , splinetype.(scl_EV  , flag), OnGrid())
    it_dEV   = interpolate!(sev.dEV , splinetype.(scl_dEV , flag), OnGrid())
    it_dEVσ  = interpolate!(sev.dEVσ, splinetype.(scl_dEVσ, flag), OnGrid())
    it_dEVψ  = interpolate!(sev.dEVψ, splinetype.(scl_dEVσ, flag), OnGrid())

    sit_EV   = Interpolations.scale(it_EV   , scl_EV...)
    sit_dEV  = Interpolations.scale(it_dEV  , scl_dEV...)
    sit_dEVσ = Interpolations.scale(it_dEVσ , scl_dEVσ...)
    sit_dEVψ = Interpolations.scale(it_dEVψ , scl_dEVσ...)

    return ItpSharedEV{T,typeof(sit_EV),typeof(sit_dEV),typeof(sit_dEVσ),TT}(sit_EV, sit_dEV, sit_dEVσ, sit_dEVψ, sev.itypes)
end

function ItpSharedEV(σ::Real, flag::Interpolations.Flag=Linear())
  global g_SharedEV
  global g_dcdp_primitives
  return ItpSharedEV(g_SharedEV, g_dcdp_primitives, σ, flag)
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
