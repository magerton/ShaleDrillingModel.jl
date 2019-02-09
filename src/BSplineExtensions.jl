# module BSplineExtensions

export prefilterByView!, serial_prefilterByView!, parallel_prefilterByView!, gradient_d, gradient_d_impl

using Interpolations
using Interpolations: iextract, itptype, DimSpec, BSplineInterpolation, prefilter!,
    gridtype, tweight, GridType, gradient_impl, count_interp_dims, gradient_coefficients, index_gen, define_indices,
    sqr, cub, InterpolationType, coordlookup, rescale_gradient, Flag

using Base.Cartesian
using Ratios


const NoPrefiltering = DimSpec{BSpline{<:Union{Linear,Constant}}}

prefilterByView!(         x::AA, Btype::Type{<:BSplineInterpolation{TWeight,N,AA,IT,GT,0}}, CI::CartesianIndex) where {TWeight,N,AA<:AbstractArray{<:Real,N},IT<:NoPrefiltering  ,GT} = nothing
function prefilterByView!(x::AA, Btype::Type{<:BSplineInterpolation{TWeight,N,AA,IT,GT,0}}, CI::CartesianIndex) where {TWeight,N,AA<:AbstractArray{<:Real,N},IT<:DimSpec{BSpline},GT}
    Nsmall = N - length(CI)
    all(map(i -> iextract(IT,i) <: NoPrefiltering, Nsmall+1:N))  || throw(error("One of indexes $(Nsmall+1:N) require prefiltering. Interpolation spec was $IT"))
    ITsmall = (map(i -> iextract(IT, i), 1:Nsmall)...,)
    @views xvw = x[ntuple((x)-> Colon(), Nsmall)...,CI]
    prefilter!(TWeight, xvw, Tuple{ITsmall...}, GT)
end

serial_prefilterByView!(         x::AA, B::BST, dims::Integer...) where {T,N,AA<:AbstractArray{T,N}, BST<:BSplineInterpolation{<:Real,N,AA,<:NoPrefiltering,  <:GridType,0} } = nothing
function serial_prefilterByView!(x::AA, B::BST, dims::Integer...) where {T,N,AA<:AbstractArray{T,N}, BST<:BSplineInterpolation{<:Real,N,AA,<:DimSpec{BSpline},<:GridType,0} }
    x === B.coefs || throw(error("x is not the coefs of B!"))
    CR = CartesianIndices(dims)
    for CI in CR
        prefilterByView!(x, BST, CI)
    end
end

parallel_prefilterByView!(         x::SharedArray{T,N}, B::BST, dims::Integer...) where {T,N, BST<:BSplineInterpolation{<:Real,N,SharedArray{T,N},<:NoPrefiltering,  <:GridType,0} } = nothing
function parallel_prefilterByView!(x::SharedArray{T,N}, B::BST, dims::Integer...) where {T,N, BST<:BSplineInterpolation{<:Real,N,SharedArray{T,N},<:DimSpec{BSpline},<:GridType,0} }
    x === B.coefs || throw(error("x is not the coefs of B!"))
    CR = CartesianIndices(dims)
    s = @sync @distributed for CI in collect(CR)
        prefilterByView!(x, BST, CI)
    end
    fetch(s)
end



function gradient_d_impl(d::Integer, itp::Type{BSplineInterpolation{T,N,TCoefs,IT,GT,Pad}}) where {T,N,TCoefs,IT<:DimSpec{BSpline},GT<:DimSpec{GridType},Pad}
    meta = Expr(:meta, :inline)
    # For each component of the gradient, alternately calculate
    # coefficients and set component
    # n = count_interp_dims(IT, N)
    exs = Array{Expr, 1}(undef, 2)
    count_interp_dims(iextract(IT, d), 1) > 0 || throw(error("dimension $d not interpolated"))
    exs[1] = gradient_coefficients(IT, N, d)
    exs[2] = :(@inbounds g = $(index_gen(IT, N)))
    gradient_exprs = Expr(:block, exs...)
    quote
        $meta
        @nexprs $N d->(x_d = xs[d])
        inds_itp = axes(itp)

        # Calculate the indices of all coefficients that will be used
        # and define fx = x - xi in each dimension
        $(define_indices(IT, N, Pad))

        $gradient_exprs

        g::T
    end
end


# gradient_d(d::Integer, itp::BSplineInterpolation, xs::Number...) = gradient_d(Val{d}, itp, xs...)


@generated function gradient_d(::Type{Val{d}}, itp::BSplineInterpolation{T,N}, xs::Number...) where {d,T,N}
    length(xs) == N || error("Can only be called with $N indexes")
    1 <= d <= N || throw(DomainError("$d ∉ 1:$N"))
    gradient_d_impl(d, itp)
end

@generated function gradient_d(dval::Type{Val{d}}, sitp::ScaledInterpolation{T,N,ITPT,IT}, xs::Number...) where {d,T,N,ITPT,IT}
    length(xs) == N || throw(DimensionMismatch("Must index into $N-dimensional scaled interpolation object with exactly $N indices (you used $(length(xs)))"))

    interp_types = length(IT.parameters) == N ? IT.parameters : tuple([IT.parameters[1] for _ in 1:N]...)
    interp_dimens = map(it -> interp_types[it] != NoInterp, 1:N)
    interp_indices = map(i -> interp_dimens[i] ? :(coordlookup(sitp.ranges[$i], xs[$i])) : :(xs[$i]), 1:N)

    quote
        g = gradient_d(dval, sitp.itp, $(interp_indices...))
        return rescale_gradient(sitp.ranges[d], g)::T
    end
end




# because of https://github.com/JuliaMath/Interpolations.jl/pull/182

fixedgradient(sitp::ScaledInterpolation{T,N,ITPT,IT,GT}, xs::Real...) where {T,N,ITPT,IT<:DimSpec{InterpolationType},GT<:DimSpec{GridType}} = fixedgradient!(Array{T}(undef,count_interp_dims(IT,N)), sitp, xs...)
fixedgradient(sitp::ScaledInterpolation{T,N,ITPT,IT,GT}, xs...      ) where {T,N,ITPT,IT<:DimSpec{InterpolationType},GT<:DimSpec{GridType}} = fixedgradient!(Array{T}(undef,count_interp_dims(IT,N)), sitp, xs...)

@generated function fixedgradient!(g, sitp::ScaledInterpolation{T,N,ITPT,IT}, xs::Number...) where {T,N,ITPT,IT}
    ndims(g) == 1 || throw(DimensionMismatch("g must be a vector (but had $(ndims(g)) dimensions)"))
    length(xs) == N || throw(DimensionMismatch("Must index into $N-dimensional scaled interpolation object with exactly $N indices (you used $(length(xs)))"))

    interp_types = length(IT.parameters) == N ? IT.parameters : tuple([IT.parameters[1] for _ in 1:N]...)
    interp_dimens = map(it -> interp_types[it] != NoInterp, 1:N)
    interp_indices = map(i -> interp_dimens[i] ? :(coordlookup(sitp.ranges[$i], xs[$i])) : :(xs[$i]), 1:N)

    quote
        length(g) == $(count_interp_dims(IT, N)) || throw(ArgumentError(string("The length of the provided gradient vector (", length(g), ") did not match the number of interpolating dimensions (", $(count_interp_dims(IT, N)), ")")))
        Interpolations.gradient!(g, sitp.itp, $(interp_indices...))
        cntr = 0
        for i = 1:N
                if $(interp_dimens)[i]
                    cntr += 1
                    g[cntr] = rescale_gradient(sitp.ranges[i], g[cntr])
                end
        end
        g
    end
end














# module end
# end


# tw = tweight(x)
# tc = tcoef(x)

# function prefilterView!(sit::ItpSharedEV, CI::CartesianIndex)
#     prefilterView!(sit.EV, CI)
#     prefilterView!(sit.dEV, CI)
#     prefilterView!(sit.dEVσ, CI)
# end
#
# function parallel_prefilterView!(sit::ItpSharedEV)
#     @distributed for CI in collect( CartesianRange(length.(sit.itypes)) )
#         prefilterView!(sit, CI)
#     end
# end



#= these are the actual functions...
https://github.com/JuliaMath/Interpolations.jl/blob/master/src/b-splines/b-splines.jl#L17-L31
https://github.com/JuliaMath/Interpolations.jl/blob/master/src/b-splines/b-splines.jl#L91-L94

Prefiltering at https://github.com/JuliaMath/Interpolations.jl/blob/master/src/b-splines/prefiltering.jl#L63-L79

interpolate!(::Type{TWeights}, A, it::IT, gt::GT) where {TWeights,IT<:DimSpec{BSpline},GT<:DimSpec{GridType}} = BSplineInterpolation(TWeights, prefilter!(TWeights, A, IT, GT), it, gt, Val{0}())
function interpolate!(A::AbstractArray, it::IT, gt::GT) where {IT<:DimSpec{BSpline},GT<:DimSpec{GridType}}
    interpolate!(tweight(A), A, it, gt)
end

struct BSplineInterpolation{T,N,TCoefs<:AbstractArray,IT<:DimSpec{BSpline},GT<:DimSpec{GridType},pad} <: AbstractInterpolation{T,N,IT,GT}
    coefs::TCoefs
end
function BSplineInterpolation(::Type{TWeights}, A::AbstractArray{Tel,N}, ::IT, ::GT, ::Val{pad}) where {N,Tel,TWeights<:Real,IT<:DimSpec{BSpline},GT<:DimSpec{GridType},pad}
    isleaftype(IT) || error("The b-spline type must be a leaf type (was $IT)")
    isleaftype(typeof(A)) || @warn "For performance reasons, consider using an array of a concrete type (typeof(A) == $(typeof(A)))"

    c = zero(TWeights)
    for _ in 2:N
        c *= c
    end
    T = Core.Inference.return_type(*, Tuple{typeof(c), Tel})

    BSplineInterpolation{T,N,typeof(A),IT,GT,pad}(A)
end
=#


# # https://github.com/JuliaMath/jl/blob/master/src/b-splines/b-splines.jl#L72-L75
# # function interpolate(::Type{TWeights}, ::Type{TC}, A, it::IT, gt::GT) where {TWeights,TC,IT<:DimSpec{BSpline},GT<:DimSpec{GridType}}
# #     Apad, Pad = prefilter(TWeights, TC, A, IT, GT)
# #     BSplineInterpolation(TWeights, Apad, it, gt, Pad)
# # end
#
# Apad, Pad = prefilter(tw, tc, x, IT, GT)
# bsit = BSplineInterpolation(tw, Apad, it, gt, Pad)
#
# # https://github.com/JuliaMath/jl/blob/master/src/b-splines/prefiltering.jl#L30-L79
# # prefilter!(::Type{TWeights}, A, ::Type{IT}, ::Type{GT}) where {TWeights, IT<:BSpline, GT<:GridType} = A
# # function prefilter(::Type{TWeights}, ::Type{TC}, A, ::Type{IT}, ::Type{GT}) where {TWeights, TC, IT<:BSpline, GT<:GridType}
# #     coefs = similar(dims->Array{TC}(dims), indices(A))
# #     prefilter!(TWeights, copy!(coefs, A), IT, GT), Val{0}()
# # end
# #
# # function prefilter(
# #     ::Type{TWeights}, ::Type{TC}, A::AbstractArray, ::Type{BSpline{IT}}, ::Type{GT}
# #     ) where {TWeights,TC,IT<:Union{Cubic,Quadratic},GT<:GridType}
# #     ret, Pad = copy_with_padding(TC, A, BSpline{IT})
# #     prefilter!(TWeights, ret, BSpline{IT}, GT), Pad
# # end
# #
# # function prefilter(
# #     ::Type{TWeights}, ::Type{TC}, A::AbstractArray, ::Type{IT}, ::Type{GT}
# #     ) where {TWeights,TC,IT<:Tuple{Vararg{Union{BSpline,NoInterp}}},GT<:DimSpec{GridType}}
# #     ret, Pad = copy_with_padding(TC, A, IT)
# #     prefilter!(TWeights, ret, IT, GT), Pad
# # end
#
# # https://github.com/JuliaMath/jl/blob/master/src/b-splines/prefiltering.jl#L15-L28
# # function copy_with_padding(::Type{TC}, A, ::Type{IT}) where {TC,IT<:DimSpec{InterpolationType}}
# #     Pad = padding(IT)
# #     indsA = indices(A)
# #     indscp, indspad = padded_index(indsA, Pad)
# #     coefs = similar(dims->Array{TC}(dims), indspad)
# #     if indspad == indsA
# #         coefs = copy!(coefs, A)
# #     else
# #         fill!(coefs, zero(TC))
# #         copy!(coefs, CartesianRange(indscp), A, CartesianRange(indsA))
# #     end
# #     coefs, Pad
# # end
#
# # https://github.com/JuliaMath/jl/blob/master/src/b-splines/quadratic.jl#L208-L226
#
# map(length, indices(x))
#
# prefiltering_system(Float64, Float64, 5, Linear(), OnCell())
#
#
# indices_removepad(indices(x), Pad)
#
# bsit.coefs === Apad
#
# Pad = padding(IT)
# indsA = indices(x)
# indscp, indspad = padded_index(indsA, Pad)
#
# inner_system_diags(Float64,2,Quadratic{Line})
#
#
#
# # https://github.com/JuliaMath/jl/blob/master/src/b-splines/prefiltering.jl#L63-L79
# # function prefilter!(
# #     ::Type{TWeights}, ret::TCoefs, ::Type{IT}, ::Type{GT}
# #     ) where {TWeights,TCoefs<:AbstractArray,IT<:Tuple{Vararg{Union{BSpline,NoInterp}}},GT<:DimSpec{GridType}}
# #     local buf, shape, retrs
# #     sz = size(ret)
# #     first = true
# #     for dim in 1:ndims(ret)
# #         it = iextract(IT, dim)
# #         if it != NoInterp
# #             M, b = prefiltering_system(TWeights, eltype(TCoefs), sz[dim], bsplinetype(it), iextract(GT, dim))
# #             if M != nothing
# #                 A_ldiv_B_md!(ret, M, ret, dim, b)
# #             end
# #         end
# #     end
# #     ret
# # end
