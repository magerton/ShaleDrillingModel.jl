using Interpolations
using Base.Test

x = SharedArray{Float64}(100)
rand!(x)
y = copy(x)

bsl = BSpline(Linear())
bsq = BSpline(Quadratic(InPlace()))
bsql = BSpline(Quadratic(Line()))
noi = NoInterp()

@test all(x .== y)
inplace_itp = interpolate!(x, (bsq, bsq, bsq), OnCell())
copy_itp = interpolate(y, (bsq, bsq, bsq), OnCell())

@show typeof(inplace_itp) typeof(copy_itp)

inplace_itp[3,3,3] - copy_itp[3,3,3]



all(x .== y)
fieldnames(itp)
typeof(itp.coefs)
size(itp.coefs)
itp.coefs === x


x.=0.0
x

itp.coefs[:,2:6,:] .- x

it = (bsl, bsq, noi)
IT = typeof(it)
gt = OnGrid()
GT = typeof(gt)
tw = Interpolations.tweight(x)
tc = Interpolations.tcoef(x)

# https://github.com/JuliaMath/Interpolations.jl/blob/master/src/b-splines/b-splines.jl#L72-L75
# function interpolate(::Type{TWeights}, ::Type{TC}, A, it::IT, gt::GT) where {TWeights,TC,IT<:DimSpec{BSpline},GT<:DimSpec{GridType}}
#     Apad, Pad = prefilter(TWeights, TC, A, IT, GT)
#     BSplineInterpolation(TWeights, Apad, it, gt, Pad)
# end

Apad, Pad = Interpolations.prefilter(tw, tc, x, IT, GT)
bsit = Interpolations.BSplineInterpolation(tw, Apad, it, gt, Pad)

# https://github.com/JuliaMath/Interpolations.jl/blob/master/src/b-splines/prefiltering.jl#L30-L79
# prefilter!(::Type{TWeights}, A, ::Type{IT}, ::Type{GT}) where {TWeights, IT<:BSpline, GT<:GridType} = A
# function prefilter(::Type{TWeights}, ::Type{TC}, A, ::Type{IT}, ::Type{GT}) where {TWeights, TC, IT<:BSpline, GT<:GridType}
#     coefs = similar(dims->Array{TC}(dims), indices(A))
#     prefilter!(TWeights, copy!(coefs, A), IT, GT), Val{0}()
# end
#
# function prefilter(
#     ::Type{TWeights}, ::Type{TC}, A::AbstractArray, ::Type{BSpline{IT}}, ::Type{GT}
#     ) where {TWeights,TC,IT<:Union{Cubic,Quadratic},GT<:GridType}
#     ret, Pad = copy_with_padding(TC, A, BSpline{IT})
#     prefilter!(TWeights, ret, BSpline{IT}, GT), Pad
# end
#
# function prefilter(
#     ::Type{TWeights}, ::Type{TC}, A::AbstractArray, ::Type{IT}, ::Type{GT}
#     ) where {TWeights,TC,IT<:Tuple{Vararg{Union{BSpline,NoInterp}}},GT<:DimSpec{GridType}}
#     ret, Pad = copy_with_padding(TC, A, IT)
#     prefilter!(TWeights, ret, IT, GT), Pad
# end

# https://github.com/JuliaMath/Interpolations.jl/blob/master/src/b-splines/prefiltering.jl#L15-L28
# function copy_with_padding(::Type{TC}, A, ::Type{IT}) where {TC,IT<:DimSpec{InterpolationType}}
#     Pad = padding(IT)
#     indsA = indices(A)
#     indscp, indspad = padded_index(indsA, Pad)
#     coefs = similar(dims->Array{TC}(dims), indspad)
#     if indspad == indsA
#         coefs = copy!(coefs, A)
#     else
#         fill!(coefs, zero(TC))
#         copy!(coefs, CartesianRange(indscp), A, CartesianRange(indsA))
#     end
#     coefs, Pad
# end

# https://github.com/JuliaMath/Interpolations.jl/blob/master/src/b-splines/quadratic.jl#L208-L226

map(length, indices(x))

Interpolations.prefiltering_system(Float64, Float64, 5, Linear(), OnCell())


Interpolations.indices_removepad(indices(x), Pad)

bsit.coefs === Apad

Pad = Interpolations.padding(IT)
indsA = indices(x)
indscp, indspad = Interpolations.padded_index(indsA, Pad)

Interpolations.inner_system_diags(Float64,2,Quadratic{Line})



# https://github.com/JuliaMath/Interpolations.jl/blob/master/src/b-splines/prefiltering.jl#L63-L79
# function prefilter!(
#     ::Type{TWeights}, ret::TCoefs, ::Type{IT}, ::Type{GT}
#     ) where {TWeights,TCoefs<:AbstractArray,IT<:Tuple{Vararg{Union{BSpline,NoInterp}}},GT<:DimSpec{GridType}}
#     local buf, shape, retrs
#     sz = size(ret)
#     first = true
#     for dim in 1:ndims(ret)
#         it = iextract(IT, dim)
#         if it != NoInterp
#             M, b = prefiltering_system(TWeights, eltype(TCoefs), sz[dim], bsplinetype(it), iextract(GT, dim))
#             if M != nothing
#                 A_ldiv_B_md!(ret, M, ret, dim, b)
#             end
#         end
#     end
#     ret
# end
