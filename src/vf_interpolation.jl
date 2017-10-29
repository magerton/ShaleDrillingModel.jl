export ItpSharedEV

struct ItpSharedEV{T,A1<:Interpolations.AbstractInterpolation{T},A2<:Interpolations.AbstractInterpolation{T},A3<:Interpolations.AbstractInterpolation{T},TT<:Tuple}
  EV::A1
  dEV::A2
  dEVσ::A3
  # dEVψ::A3
  itypes::TT
end

function ItpSharedEV(σ::Real, ψflag::Flag=Quadratic(InPlace()) )
  global g_SharedEV
  global g_dcdp_primitives
  return ItpSharedEV(g_SharedEV, g_dcdp_primitives, σ, ψflag)
end

@GenGlobal g_ItpSharedEV


function ItpSharedEV(sev::SharedEV{T,N,N2,TT}, p::dcdp_primitives, σ::Real=1.0; ψflag::Flag=Quadratic(InPlace())) where {T,N,N2,TT}

    # dimensions of things
    ntyp = length(sev.itypes)
    nθ = _nθt(p)
    nS = _nS(p)
    nSexp1 = _nSexp(p)

    # form spline specifications
    splinetype(r::Union{StepRange{<:AbstractFloat},StepRangeLen{<:AbstractFloat}}) = BSpline(Linear())
    splinetype(r::AbstractVector) = NoInterp()

    it_z     = splinetype.(p.zspace)
    it_itype = splinetype.(sev.itypes)

    it_EV   = (it_z..., BSpline(ψflag), NoInterp(),             it_itype...)
    it_dEV  = (it_z..., BSpline(ψflag), NoInterp(), NoInterp(), it_itype...)
    it_dEVσ = (it_z..., BSpline(ψflag), NoInterp(),             it_itype...)

    # form UNFILTERED spline objects.
    # These specify the SharedArray as the coef field and add type information about spline interpolation
    # if quadtratic, we will have to solve a system of linear equations later to interpolate
    itp_EV    = BSplineInterpolation(tweight(sev.EV  ), sev.EV  , it_EV  , OnCell(), Val{0}())
    itp_dEV   = BSplineInterpolation(tweight(sev.dEV ), sev.dEV , it_dEV , OnCell(), Val{0}())
    itp_dEVσ  = BSplineInterpolation(tweight(sev.dEVσ), sev.dEVσ, it_dEVσ, OnCell(), Val{0}())

    # information for how to scale the interpolation object
    scalegrid(x::Range{S}) where {S<:AbstractFloat} = x
    scalegrid(x::Integer) = Base.OneTo(x)
    scalegrid(x::AbstractVector) = Base.OneTo(length(x))

    # scaled interpolation
    scl_EV   = scalegrid.((p.zspace..., _ψspace(p,σ),         nS, sev.itypes...))
    scl_dEV  = scalegrid.((p.zspace..., _ψspace(p,σ), nθ,     nS, sev.itypes...))
    scl_dEVσ = scalegrid.((p.zspace..., _ψspace(p,σ),     nSexp1, sev.itypes...))

    sit_EV   = scale(itp_EV   , scl_EV...)
    sit_dEV  = scale(itp_dEV  , scl_dEV...)
    sit_dEVσ = scale(itp_dEVσ , scl_dEVσ...)
    # sit_dEVψ = Interpolations.scale(it_dEVψ , scl_dEVσ...)

    return ItpSharedEV{T,typeof(sit_EV),typeof(sit_dEV),typeof(sit_dEVσ),TT}(sit_EV, sit_dEV, sit_dEVσ, sev.itypes)
end


for typ in (:parallel, :serial)
    fun = Symbol("$(typ)_prefilterByView!")
    @eval begin
        function ($fun)(sev::SharedEV{T,N,N2}, isev::ItpSharedEV, dograd::Bool=true) where {T,N,N2}
            Ntyps = length(sev.itypes)
            ($fun)(sev.EV, isev.EV.itp,        length.(sev.itypes)...)
            if dograd
                ($fun)(sev.dEV, isev.dEV.itp,   length.(sev.itypes)...)
                ($fun)(sev.dEVσ, isev.dEVσ.itp, length.(sev.itypes)...)
            end
        end
    end
end

# function parallel_prefilterByView!(sev::SharedEV{T,N,N2}, isev::ItpSharedEV)
#     Ntyps = length.(sev.itypes)
#     parallel_prefilterByView!(sev.EV, isev.EV,     N- Ntyps+1:N...)
#     parallel_prefilterByView!(sev.dEV, isev.dEV,   N2-Ntyps+1:N...)
#     parallel_prefilterByView!(sev.dEVσ, isev.dEVσ, N- Ntyps+1:N...)
# end
#
# function parallel_prefilterByView!(sev::SharedEV{T,N,N2}, isev::ItpSharedEV)
#     Ntyps = length.(sev.itypes)
#     parallel_prefilterByView!(sev.EV, isev.EV,     N- Ntyps+1:N...)
#     parallel_prefilterByView!(sev.dEV, isev.dEV,   N2-Ntyps+1:N...)
#     parallel_prefilterByView!(sev.dEVσ, isev.dEVσ, N- Ntyps+1:N...)
# end

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
