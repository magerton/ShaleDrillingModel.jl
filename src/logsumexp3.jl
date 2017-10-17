export logsumexp3!, logsumexp_and_softmax3!, softmax3!

function logsumexp3!(lse::AbstractMatrix, tmp::AbstractMatrix, x::AbstractArray{T,3}) where {T<:Real}
    nz, nψ, nd = size(x)
    (nz, nψ) ==  size(lse) == size(tmp) || throw(DimensionMismatch())

    maximum!(reshape(tmp, nz,nψ,1), x)
    lse .= exp.(@view(x[:,:,1]) .- tmp)
    @inbounds for k in 2:nd
        lse .+= exp.(@view(x[:,:,k]) .- tmp)
    end
    lse .= log.(lse) .+ tmp
end


function softmax3!(q::AbstractArray{T,3}, lse::AbstractArray{T,2}, tmp::AbstractArray{T,2}, x::AbstractArray{T,3}) where {T}
    nz, nψ, nd = size(x)
    (nz, nψ, nd) == size(q) || throw(DimensionMismatch())
    (nz, nψ) ==  size(lse) == size(tmp) || throw(DimensionMismatch())

    maximum!(reshape(tmp, nz,nψ,1), x)
    lse .= (q[:,:,1] .= exp.(@view(x[:,:,1]) .- tmp))
    @inbounds for k in 2:size(x,3)
        lse .+= (q[:,:,k] .= exp.(@view(x[:,:,k]) .- tmp))
    end
    q ./= lse
end


# update q as Pr(x[:,:,1])
function softmax3!(q::AbstractArray{T,2}, lse::AbstractArray{T,2}, tmp::AbstractArray{T,2}, x::AbstractArray{T,3}) where {T}
    nz, nψ, nd = size(x)
    (nz, nψ) ==  size(lse) == size(tmp) == size(q) || throw(DimensionMismatch())

    maximum!(reshape(tmp, nz,nψ,1), x)
    lse .= (q .= exp.(@view(x[:,:,1]) .- tmp))
    @inbounds for k in 2:size(x,3)
        lse .+= exp.(@view(x[:,:,k]) .- tmp)
    end
    q ./= lse
end


# updates updates x to Pr(x)
softmax3!(x::AbstractArray{T,3}, lse::AbstractArray{T,2}, tmp::AbstractArray{T,2}) where {T} = softmax3!(x, lse, tmp, x)

# does not change x
function logsumexp_and_softmax3!(lse::AbstractArray{T,2}, q::AbstractArray{T}, tmp::AbstractArray{T,2}, x::AbstractArray{T,3}) where {T}
    softmax3!(q, lse, tmp, x)
    lse .= log.(lse) .+ tmp
end

# updates x to Pr(x)
function logsumexp_and_softmax3!(lse::AbstractArray{T,2}, x::AbstractArray{T,3}, tmp::AbstractArray{T,2}) where {T}
    logsumexp_and_softmax3!(lse, x, tmp, x)
end
