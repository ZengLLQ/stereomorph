# StereoMorph — Types & Molecular Representation Module

module Types

using LinearAlgebra
using Printf

export Hybridization, SP3, SP2, SP, UNKNOWN_HYB
export ChiralSign, R_CHIRAL, S_CHIRAL, ACHIRAL
export Atom, Bond, ChiralCenter, PlanarSystem, Molecule, Conformation
export ELEMENTS, get_element_info
export shortest_path_matrix, find_rings
export make_methane, make_ethane, make_propane, make_butane, make_cyclohexane
export make_benzene, make_aspirin, make_alanine, make_ibuprofen

@enum Hybridization SP3 SP2 SP UNKNOWN_HYB
@enum ChiralSign R_CHIRAL S_CHIRAL ACHIRAL

"""
    Atom

Represents a single atom in a molecular topology.
"""
struct Atom
    index::Int
    symbol::String
    atomic_number::Int
    mass::Float64
    covalent_radius::Float64
    vdw_radius::Float64
    hybridization::Hybridization
    formal_charge::Int
end

"""
    Bond

Represents a chemical covalent bond between two atoms.
"""
struct Bond
    atom1::Int
    atom2::Int
    order::Float64      # 1.0 = single, 1.5 = aromatic, 2.0 = double, 3.0 = triple
    aromatic::Bool
end

"""
    ChiralCenter

Represents a tetrahedral stereocenter with 4 designated neighbors and target spatial parity.
"""
struct ChiralCenter
    center::Int
    neighbors::NTuple{4, Int}
    target_sign::Float64   # +1.0 for (R) or clockwise, -1.0 for (S) or counter-clockwise
    name::String
end

"""
    PlanarSystem

Represents a collection of atoms that must lie in a common geometric plane (e.g. aromatic rings, amides).
"""
struct PlanarSystem
    atoms::Vector{Int}
    description::String
end

"""
    Molecule

Complete chemical topology including atoms, bonds, stereocenters, and planar ring systems.
"""
struct Molecule
    name::String
    atoms::Vector{Atom}
    bonds::Vector{Bond}
    adj_list::Vector{Vector{Int}}
    bond_orders::Dict{Tuple{Int,Int}, Float64}
    chiral_centers::Vector{ChiralCenter}
    planar_systems::Vector{PlanarSystem}

    function Molecule(name::AbstractString, atoms::Vector{Atom}, bonds::Vector{Bond},
                      chiral_centers::Vector{ChiralCenter}=ChiralCenter[],
                      planar_systems::Vector{PlanarSystem}=PlanarSystem[])
        n = length(atoms)
        adj = [Int[] for _ in 1:n]
        orders = Dict{Tuple{Int,Int}, Float64}()

        for b in bonds
            push!(adj[b.atom1], b.atom2)
            push!(adj[b.atom2], b.atom1)
            orders[(min(b.atom1, b.atom2), max(b.atom1, b.atom2))] = b.order
        end

        new(String(name), atoms, bonds, adj, orders, chiral_centers, planar_systems)
    end
end

"""
    Conformation

3D Cartesian geometry (3 x N matrix) along with energy, quality, and convergence metadata.
"""
struct Conformation
    coords::Matrix{Float64}         # 3 x N coordinates in Angstroms
    energy::Float64                 # Penalty energy (kcal/mol or arbitrary unit)
    converged::Bool                 # Did optimization converge
    embedding_quality::Float64      # Fraction of positive eigenvalues in 3D projection
    violations::Int                 # Count of distance/chiral bounds violations
end

# Standard elemental properties: (atomic_number, mass, covalent_radius, vdw_radius)
const ELEMENTS = Dict{String, Tuple{Int, Float64, Float64, Float64}}(
    "H"  => (1,   1.008, 0.31, 1.20),
    "C"  => (6,  12.011, 0.76, 1.70),
    "N"  => (7,  14.007, 0.71, 1.55),
    "O"  => (8,  15.999, 0.66, 1.52),
    "F"  => (9,  18.998, 0.57, 1.47),
    "Na" => (11, 22.990, 1.66, 2.27),
    "Mg" => (12, 24.305, 1.41, 1.73),
    "P"  => (15, 30.974, 1.07, 1.80),
    "S"  => (16, 32.060, 1.05, 1.80),
    "Cl" => (17, 35.450, 1.02, 1.75),
    "Br" => (35, 79.904, 1.20, 1.85),
    "I"  => (53, 126.90, 1.39, 1.98)
)

function get_element_info(sym::AbstractString)
    sym_str = String(sym)
    if haskey(ELEMENTS, sym_str)
        return ELEMENTS[sym_str]
    else
        return (0, 12.0, 0.77, 1.70)
    end
end

"""
    shortest_path_matrix(mol::Molecule)::Matrix{Int}

Computes the topological graph distance (bond count) between all pairs of atoms via BFS.
"""
function shortest_path_matrix(mol::Molecule)::Matrix{Int}
    n = length(mol.atoms)
    dist = fill(999, n, n)
    for i in 1:n
        dist[i, i] = 0
    end

    for start in 1:n
        queue = [start]
        visited = falses(n)
        visited[start] = true
        dist[start, start] = 0

        while !isempty(queue)
            curr = popfirst!(queue)
            d_curr = dist[start, curr]
            for neighbor in mol.adj_list[curr]
                if !visited[neighbor]
                    visited[neighbor] = true
                    dist[start, neighbor] = d_curr + 1
                    push!(queue, neighbor)
                end
            end
        end
    end
    return dist
end

"""
    find_rings(mol::Molecule)::Vector{Vector{Int}}

Detects fundamental cycles (rings) in the molecular graph using DFS cycle extraction.
"""
function find_rings(mol::Molecule)::Vector{Vector{Int}}
    n = length(mol.atoms)
    visited = falses(n)
    parent = fill(0, n)
    rings = Vector{Int}[]

    function dfs(u::Int, p::Int)
        visited[u] = true
        parent[u] = p
        for v in mol.adj_list[u]
            if v == p
                continue
            end
            if visited[v]
                # Found cycle
                cycle = Int[u]
                curr = parent[u]
                while curr != 0 && curr != v
                    push!(cycle, curr)
                    curr = parent[curr]
                end
                push!(cycle, v)
                # Keep canonical representation (sort minimum cyclic rotation)
                if length(cycle) >= 3 && length(cycle) <= 8
                    push!(rings, cycle)
                end
            else
                dfs(v, u)
            end
        end
    end

    for i in 1:n
        if !visited[i]
            dfs(i, 0)
        end
    end

    # Deduplicate rings by sorting atom sets
    unique_rings = Vector{Int}[]
    seen_sets = Set{Set{Int}}()
    for r in rings
        s = Set(r)
        if !(s in seen_sets)
            push!(seen_sets, s)
            push!(unique_rings, r)
        end
    end

    return unique_rings
end

# -------------------------------------------------------------
# Pre-built benchmark molecular topologies
# -------------------------------------------------------------

function make_methane()::Molecule
    c_info = get_element_info("C")
    h_info = get_element_info("H")
    atoms = [
        Atom(1, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(2, "H", h_info[1], h_info[2], h_info[3], h_info[4], SP3, 0),
        Atom(3, "H", h_info[1], h_info[2], h_info[3], h_info[4], SP3, 0),
        Atom(4, "H", h_info[1], h_info[2], h_info[3], h_info[4], SP3, 0),
        Atom(5, "H", h_info[1], h_info[2], h_info[3], h_info[4], SP3, 0),
    ]
    bonds = [
        Bond(1, 2, 1.0, false),
        Bond(1, 3, 1.0, false),
        Bond(1, 4, 1.0, false),
        Bond(1, 5, 1.0, false),
    ]
    Molecule("Methane", atoms, bonds)
end

function make_ethane()::Molecule
    c_info = get_element_info("C")
    h_info = get_element_info("H")
    atoms = [
        Atom(1, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(2, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(3, "H", h_info[1], h_info[2], h_info[3], h_info[4], SP3, 0),
        Atom(4, "H", h_info[1], h_info[2], h_info[3], h_info[4], SP3, 0),
        Atom(5, "H", h_info[1], h_info[2], h_info[3], h_info[4], SP3, 0),
        Atom(6, "H", h_info[1], h_info[2], h_info[3], h_info[4], SP3, 0),
        Atom(7, "H", h_info[1], h_info[2], h_info[3], h_info[4], SP3, 0),
        Atom(8, "H", h_info[1], h_info[2], h_info[3], h_info[4], SP3, 0),
    ]
    bonds = [
        Bond(1, 2, 1.0, false),
        Bond(1, 3, 1.0, false),
        Bond(1, 4, 1.0, false),
        Bond(1, 5, 1.0, false),
        Bond(2, 6, 1.0, false),
        Bond(2, 7, 1.0, false),
        Bond(2, 8, 1.0, false),
    ]
    Molecule("Ethane", atoms, bonds)
end

function make_propane()::Molecule
    c_info = get_element_info("C")
    atoms = [
        Atom(1, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(2, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(3, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
    ]
    bonds = [
        Bond(1, 2, 1.0, false),
        Bond(2, 3, 1.0, false),
    ]
    Molecule("Propane", atoms, bonds)
end

function make_butane()::Molecule
    c_info = get_element_info("C")
    atoms = [
        Atom(1, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(2, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(3, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(4, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
    ]
    bonds = [
        Bond(1, 2, 1.0, false),
        Bond(2, 3, 1.0, false),
        Bond(3, 4, 1.0, false),
    ]
    Molecule("Butane", atoms, bonds)
end

function make_cyclohexane()::Molecule
    c_info = get_element_info("C")
    atoms = [Atom(i, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0) for i in 1:6]
    bonds = [
        Bond(1, 2, 1.0, false),
        Bond(2, 3, 1.0, false),
        Bond(3, 4, 1.0, false),
        Bond(4, 5, 1.0, false),
        Bond(5, 6, 1.0, false),
        Bond(6, 1, 1.0, false),
    ]
    Molecule("Cyclohexane", atoms, bonds)
end

function make_benzene()::Molecule
    c_info = get_element_info("C")
    atoms = [Atom(i, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0) for i in 1:6]
    bonds = [
        Bond(1, 2, 1.5, true),
        Bond(2, 3, 1.5, true),
        Bond(3, 4, 1.5, true),
        Bond(4, 5, 1.5, true),
        Bond(5, 6, 1.5, true),
        Bond(6, 1, 1.5, true),
    ]
    planar = [PlanarSystem([1, 2, 3, 4, 5, 6], "Aromatic Benzene Ring")]
    Molecule("Benzene", atoms, bonds, ChiralCenter[], planar)
end

function make_aspirin()::Molecule
    c_info = get_element_info("C")
    o_info = get_element_info("O")
    # Acetylsalicylic acid: 9 carbons, 4 oxygens = 13 heavy atoms
    atoms = [
        # Benzene ring (1-6)
        Atom(1, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(2, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(3, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(4, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(5, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(6, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        # Carboxylic acid at C1: C7(=O8)-O9
        Atom(7, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(8, "O", o_info[1], o_info[2], o_info[3], o_info[4], SP2, 0),
        Atom(9, "O", o_info[1], o_info[2], o_info[3], o_info[4], SP3, 0),
        # Ester group at C2: -O10-C11(=O12)-C13
        Atom(10, "O", o_info[1], o_info[2], o_info[3], o_info[4], SP3, 0),
        Atom(11, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(12, "O", o_info[1], o_info[2], o_info[3], o_info[4], SP2, 0),
        Atom(13, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
    ]
    bonds = [
        # Ring bonds
        Bond(1, 2, 1.5, true),
        Bond(2, 3, 1.5, true),
        Bond(3, 4, 1.5, true),
        Bond(4, 5, 1.5, true),
        Bond(5, 6, 1.5, true),
        Bond(6, 1, 1.5, true),
        # Carboxylic acid
        Bond(1, 7, 1.0, false),
        Bond(7, 8, 2.0, false),
        Bond(7, 9, 1.0, false),
        # Ester
        Bond(2, 10, 1.0, false),
        Bond(10, 11, 1.0, false),
        Bond(11, 12, 2.0, false),
        Bond(11, 13, 1.0, false),
    ]
    planar = [
        PlanarSystem([1, 2, 3, 4, 5, 6], "Benzene Core"),
        PlanarSystem([1, 7, 8, 9], "Carboxylic Acid"),
        PlanarSystem([10, 11, 12, 13], "Acetate Ester")
    ]
    Molecule("Aspirin", atoms, bonds, ChiralCenter[], planar)
end

function make_alanine(chiral_type::Symbol=:L)::Molecule
    c_info = get_element_info("C")
    n_info = get_element_info("N")
    o_info = get_element_info("O")
    h_info = get_element_info("H")

    # L-Alanine / D-Alanine:
    # 1: C_alpha (chiral center)
    # 2: N (amino)
    # 3: C (carboxyl)
    # 4: C (methyl side chain)
    # 5: H (alpha hydrogen)
    # 6: O (carbonyl)
    # 7: O (hydroxyl)
    atoms = [
        Atom(1, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(2, "N", n_info[1], n_info[2], n_info[3], n_info[4], SP3, 0),
        Atom(3, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(4, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(5, "H", h_info[1], h_info[2], h_info[3], h_info[4], SP3, 0),
        Atom(6, "O", o_info[1], o_info[2], o_info[3], o_info[4], SP2, 0),
        Atom(7, "O", o_info[1], o_info[2], o_info[3], o_info[4], SP3, 0),
    ]
    bonds = [
        Bond(1, 2, 1.0, false),
        Bond(1, 3, 1.0, false),
        Bond(1, 4, 1.0, false),
        Bond(1, 5, 1.0, false),
        Bond(3, 6, 2.0, false),
        Bond(3, 7, 1.0, false),
    ]

    target_sign = (chiral_type == :L) ? 1.0 : -1.0
    chiral = [ChiralCenter(1, (2, 3, 4, 5), target_sign, string(chiral_type, "-Alanine Cα"))]
    planar = [PlanarSystem([1, 3, 6, 7], "Carboxyl Plane")]

    Molecule(string(chiral_type, "-Alanine"), atoms, bonds, chiral, planar)
end

function make_ibuprofen()::Molecule
    c_info = get_element_info("C")
    o_info = get_element_info("O")
    h_info = get_element_info("H")

    # (S)-Ibuprofen: 13 carbons, 2 oxygens = 15 heavy atoms
    # 1-6: aromatic ring
    # C1 connected to isobutyl: -CH2(7)-CH(8)(CH3(9))-CH3(10)
    # C4 connected to propionic acid: -CH(11)(CH3(12))-C(13)(=O(14))-O(15)
    # Chiral center at C11 with neighbors: C4 (ring), C13 (acid), C12 (methyl), and implicit/explicit H16
    atoms = [
        Atom(1,  "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(2,  "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(3,  "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(4,  "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(5,  "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(6,  "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        # Isobutyl
        Atom(7,  "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(8,  "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(9,  "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(10, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        # Chiral branch
        Atom(11, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(12, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP3, 0),
        Atom(13, "C", c_info[1], c_info[2], c_info[3], c_info[4], SP2, 0),
        Atom(14, "O", o_info[1], o_info[2], o_info[3], o_info[4], SP2, 0),
        Atom(15, "O", o_info[1], o_info[2], o_info[3], o_info[4], SP3, 0),
        Atom(16, "H", h_info[1], h_info[2], h_info[3], h_info[4], SP3, 0),
    ]
    bonds = [
        Bond(1, 2, 1.5, true),
        Bond(2, 3, 1.5, true),
        Bond(3, 4, 1.5, true),
        Bond(4, 5, 1.5, true),
        Bond(5, 6, 1.5, true),
        Bond(6, 1, 1.5, true),
        # Isobutyl bonds
        Bond(1, 7, 1.0, false),
        Bond(7, 8, 1.0, false),
        Bond(8, 9, 1.0, false),
        Bond(8, 10, 1.0, false),
        # Propionic acid branch
        Bond(4, 11, 1.0, false),
        Bond(11, 12, 1.0, false),
        Bond(11, 13, 1.0, false),
        Bond(11, 16, 1.0, false),
        Bond(13, 14, 2.0, false),
        Bond(13, 15, 1.0, false),
    ]
    chiral = [ChiralCenter(11, (4, 13, 12, 16), -1.0, "(S)-Ibuprofen C11")]
    planar = [PlanarSystem([1, 2, 3, 4, 5, 6], "Benzene Core")]
    Molecule("Ibuprofen", atoms, bonds, chiral, planar)
end

end # module Types
