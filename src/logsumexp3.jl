export logsumexp3!, logsumexp_and_softmax3!, softmax3!, logsumexp_and_softmax!, logsumexp_and_softmax


"""
    logsumexp_and_softmax!(r, x)

Set `r` = softmax(x) and return `logsumexp(x)`.
"""
function logsumexp_and_softmax!(r::AbstractArray{T}, x::AbstractArray{T}) where {T<:Float64}
    n = length(x)
    length(r) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    isempty(x) && return -T(Inf)

    u = maximum(x)                                       # max value used to re-center
    abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values

    s = zero(T)
    @inbounds @simd for i in 1:n
        s += ( r[i] = exp(x[i] - u) )
    end
    invs = one(T)/s
    r .*= invs
    return log(s) + u
end


logsumexp_and_softmax!(x::AbstractArray{T}) where {T<:Float64}= logsumexp_and_softmax!(x, x)



function logsumexp3!(lse::AbstractMatrix, tmp::AbstractMatrix, x::AbstractArray{T,3}) where {T<:Float64}
    nz, nψ, nd = size(x)
    (nz, nψ) ==  size(lse) == size(tmp) || throw(DimensionMismatch())

    maximum!(reshape(tmp, nz,nψ,1), x)
    @views lse .= exp.(x[:,:,1] .- tmp)
    @inbounds for k in 2:nd
        @views lse .+= exp.(x[:,:,k] .- tmp)
    end
    lse .= log.(lse) .+ tmp
end


function softmax3!(q::AbstractArray{T,3}, lse::AbstractArray{T,2}, tmp::AbstractArray{T,2}, x::AbstractArray{T,3}) where {T<:Float64}
    nz, nψ, nd = size(x)
    (nz, nψ, nd) == size(q) || throw(DimensionMismatch())
    (nz, nψ) ==  size(lse) == size(tmp) || throw(DimensionMismatch())

    maximum!(reshape(tmp, nz,nψ,1), x)
    @views lse .= (q[:,:,1] .= exp.(x[:,:,1] .- tmp))
    @inbounds for k in 2:size(x,3)
        @views lse .+= (q[:,:,k] .= exp.(x[:,:,k] .- tmp))
    end
    q ./= lse
end


# update q as Pr(x[:,:,1])
function softmax3!(q::AbstractArray{T,2}, lse::AbstractArray{T,2}, tmp::AbstractArray{T,2}, x::AbstractArray{T,3}) where {T<:Float64}
    nz, nψ, nd = size(x)
    (nz, nψ) ==  size(lse) == size(tmp) == size(q) || throw(DimensionMismatch())

    maximum!(reshape(tmp, nz,nψ,1), x)
    @views lse .= (q .= exp.(x[:,:,1] .- tmp))
    @inbounds for k in 2:size(x,3)
        @views lse .+= exp.(x[:,:,k] .- tmp)
    end
    q ./= lse
end


# updates updates x to Pr(x)
softmax3!(x::AbstractArray{T,3}, lse::AbstractArray{T,2}, tmp::AbstractArray{T,2}) where {T<:Float64} = softmax3!(x, lse, tmp, x)

# does not change x
function logsumexp_and_softmax3!(lse::AbstractArray{T,2}, q::AbstractArray{T}, tmp::AbstractArray{T,2}, x::AbstractArray{T,3}) where {T<:Float64}
    softmax3!(q, lse, tmp, x)
    lse .= log.(lse) .+ tmp
end

# updates x to Pr(x)
function logsumexp_and_softmax3!(lse::AbstractArray{T,2}, x::AbstractArray{T,3}, tmp::AbstractArray{T,2}) where {T<:Float64}
    logsumexp_and_softmax3!(lse, x, tmp, x)
end
