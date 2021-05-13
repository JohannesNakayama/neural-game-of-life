using NPZ
using Random
using IterTools

function generate_training_data(grid_dims, n)
    features = generate_features(grid_dims, n)
    labels = generate_labels(features)
    features = convert_to_tensor(features)
    labels = convert_to_tensor(labels)
    return features, labels
end

function generate_features(grid_dims::Tuple{Int, Int}, n::Int)
    a, b = grid_dims
    features = [Matrix(Int.(bitrand((a, b)))) for i in 1:n]
    return features
end

function generate_labels(features)
    labels = [
        Int.(next_world_state(i))
        for i in features
    ]
    return labels
end

function convert_to_tensor(matrix_array)
    converted_data = [
        [matrix_array[j][1, :] for j in 1:size(matrix_array[i])[1]]
        for i in 1:length(matrix_array)
    ]
    return converted_data
end

function next_world_state(world::Matrix)
    next_state = zeros(Bool, size(world)[1], size(world)[2])
    for i in 1:size(world)[1], j in 1:size(world)[2]
        next_state[i, j] = next_cell_state((i, j), world)
    end
    return next_state
end

function next_cell_state(cell::Tuple{Int, Int}, world::Matrix)
    i, j = cell
    neighborhood = get_neighborhood(cell, world)
    n_alive_neighbors = sum([world[i, j] for (i, j) in neighborhood])
    current_state = world[i, j]
    next_state = life_rule(current_state, n_alive_neighbors)
    return next_state
end

function get_neighborhood(cell::Tuple{Int, Int}, world::Matrix)
    i, j = cell
    dim_1_surrounding = [i - 1, i, i + 1]
    dim_2_surrounding = [j - 1, j, j + 1]
    indices = [
        p 
        for p in product(dim_1_surrounding, dim_2_surrounding) 
        if p != (i, j)
    ]
    indices = [
        i 
        for i in indices 
        if ((i[1] != 0) & 
            (i[2] != 0) & 
            (i[1] <= size(world)[1]) & 
            (i[2] <= size(world)[2]))
    ]
    return indices
end

function life_rule(current_state::Number, n_alive_neighbors::Number)
    alive_condition = (
        ((current_state == 1) & (n_alive_neighbors in [2, 3])) |  # survival condition
        ((current_state == 0) & (n_alive_neighbors == 3))  # reproduction condition
    )
    if alive_condition
        next_state = 1
    else
        next_state = 0
    end 
    return next_state
end


features, labels = generate_training_data((10, 10), 1000)

features = generate_features((10, 10), 1000)
labels = generate_labels(features)

# TODO: create dataset with HDF5
using HDF5



