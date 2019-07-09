#
# Cache for intermediate results.
#

struct Cache{T<:Number}
    elts::Vector{AbstractArray{T}}
end

"""
    is_in_cache(cache::Cache{T}, L::Int)

Check whether a vector of size L is stored in cache.
"""
function is_in_cache(cache::Cache{T}, L::Int) where T<:Number
    if any(length.(cache.elts) .== L)
        return findfirst(length.(cache.elts) .== L)
    else
        return 0
    end
end

"""
    update_cache(cache::Cache{T}, M::AbstractArray{T}) where T<:Number

Update cache with new preallocated array M if no array with the same length is
already stored.
"""
function update_cache(cache::Cache{T}, M::AbstractArray{T}) where T<:Number
    if is_in_cache(cache, length(M)) == 0
        push!(cache.elts, M)
    end
end
