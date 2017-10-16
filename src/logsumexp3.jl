function logsumexp3!(lse::AbstractArray, tmp::AbstractArray, x::AbstractArray)
    size(lse) == size(tmp) || throw(DimensionMismatch())
    tmpA = reshape(tmp, length.(Base.reduced_indices(x,3)))

    maximum!(tmpA, x)
    lse .= exp.(@view(x[:,:,1]) .- tmp)
    for k in 2:size(x,3)
        lse .+= exp.(@view(x[:,:,k]) .- tmp)
    end
    lse .= log.(lse) .+ tmp
end


function logsumexp_and_softmax3!(lse::AbstractArray{T,2}, q::AbstractArray{T,3}, tmp::AbstractArray{T,2}, x::AbstractArray{T,3}) where {T}
    size(q) == size(x) || throw(DimensionMismatch())
    size(lse) == size(tmp) || throw(DimensionMismatch())
    tmpA = reshape(tmp, length.(Base.reduced_indices(x,3)))

    maximum!(tmpA, x)
    lse .= (q[:,:,1] .= exp.(@view(x[:,:,1]) .- tmp))
    for k in 2:size(x,3)
        lse .+= (q[:,:,k] .= exp.(@view(x[:,:,k]) .- tmp))
    end
    q ./= lse
    lse .= log.(lse) .+ tmp
end

function logsumexp_and_softmax3!(lse::AbstractArray{T,2}, q::AbstractArray{T,2}, tmp::AbstractArray{T,2}, x::AbstractArray{T,3}) where {T}
    size(q) == size(lse) == size(tmp) || throw(DimensionMismatch())
    tmpA = reshape(tmp, length.(Base.reduced_indices(x,3)))

    maximum!(tmpA, x)
    lse .= (q .= exp.(@view(x[:,:,1]) .- tmp))
    for k in 2:size(x,3)
        lse .+= exp.(@view(x[:,:,k]) .- tmp)
    end
    q ./= lse
    lse .= log.(lse) .+ tmp
end
