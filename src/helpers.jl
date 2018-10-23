export peturb

""""
    ensure_diagonal(Π::M)

If `M<:SparseMatrixCSC`, ensure that diagonal has all entries (even if zero).
"""
function ensure_diagonal(Π::M) where {T<:Number, M<:SparseMatrixCSC{T}}
    n = size(Π, 1)
    A = Π +  SparseMatrixCSC{T}(I, n, n)
    typeof(A) == M || throw(error("type of Πnew != typeof(Π)"))
    Arows = rowvals(A)
    Avals = nonzeros(A)
    for j = 1:n
       for i in nzrange(A, j)
          j == Arows[i]  &&  (Avals[i] -= one(T))
       end
    end
    return A
end

ensure_diagonal(Π::M) where {T<:Number, M<:Matrix{T}} = Π


peturb(x::T) where {T} = max( abs(x), one(T) ) * cbrt(eps(T))
