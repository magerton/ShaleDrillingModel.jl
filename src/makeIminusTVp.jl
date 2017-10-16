export update_IminusTVp!


# dense version
function update_IminusTVp!(IminusTVp::Matrix{T}, Π::Matrix{T}, β::T, q0::AbstractVector{T}) where {T<:AbstractFloat}
  n, m = size(IminusTVp)
  n == m || throw(error(DimensionMismatch()))
  (n, n,) == size(Π) || throw(error(DimensionMismatch()))

  # make_eye!(IminusTVp)
  IminusTVp .= zero(T)
  @inbounds for j = 1:n
    for i = 1:n
      IminusTVp[i,j] =       i == j     ?     one(T) - Π[i, j] * β * q0[j]  :       - Π[i, j] * β * q0[j]
    end
  end
end


# sparse version
function update_IminusTVp!(IminusTVp::SparseMatrixCSC{T}, Π::SparseMatrixCSC{T}, β::T, q0::AbstractVector{T}) where {T<:AbstractFloat}
  n, m = size(IminusTVp)
  n == m             || throw(error(DimensionMismatch()))
  (n, n,) == size(Π) || throw(error(DimensionMismatch()))

  Π_rows = rowvals(Π)
  Π_vals = nonzeros(Π)
  IminusTVp_rows = rowvals( IminusTVp)
  IminusTVp_vals = nonzeros(IminusTVp)

  # consider eliminmating this check?
  length(Π_rows) == length(IminusTVp_rows) == length(Π_vals) == length(IminusTVp_vals) || throw(error("Π and IminusTVp not same pattern"))

  @inbounds for j = 1:n
    for nzi in nzrange(IminusTVp, j)
      IminusTVp_vals[nzi] =  j == IminusTVp_rows[nzi]  ?  one(T) - Π_vals[nzi] * β * q0[j]   :   - Π_vals[nzi] * β * q0[j]
    end
  end

end
