Utilities and accelerated functions designed for use by the McFACTS team at CUNY.

Still under early alpha development. Modules and functions that are not explicitly documented here are not yet ready for use.

## Merge Tree
For M. McCarthy's black hole merge tracking feature.

```python
import mcfast
from mcfast import MergeForest

# creates a forest of all black holes
forest = MergeForest("./data/", "galaxy_state_*")

bh_uuid: str = "2b422064-0a84-4687-a542-395dcb61cd4f"

# get the immediate child of a given UUID, if the UUID is valid
descendant_uuid = forest.get_descendant(bh_uuid)

# get the immediate parents of a given UUID, if the UUID is valid
(parent1_uuid, parent2_uuid) = forest.get_parents(bh_uuid)

# get the full ancestry list of a given UUID (in DFS order), 
# assuming the UUID has ancestors
ancestor_list = forest.get_ancestors(bh_uuid)


# get the full list of black holes between the UUID and 
# the final product (root)
descent_path = forest.get_lineage_to_root(bh_uuid)

# get the generation of a given UUID, where an initialized BH is generation 0
# note: implementation doesn't fully line up with existing
# reference implementation, under review
generation = forest.get_generation(bh_uuid)

# get all 'root' black holes (products remaining at the final tick)
roots = forest.roots()

# get all 'leaf' black holes (initialized BHs with no ancestors)
leaves = forest.leaves()

# get all 'singleton' black holes (neither ancestors nor children)
singletons = forest.singletons()

if forest.contains(bh_uuid):
    print("It exists!")
else:
    print("It doesn't exist!")

# get total number of nodes in the 'forest'
total = len(forest)

# serialize and deserialize with pickle
import pickle
 
# Save now
with open("forest.pkl", "wb") as f:
    pickle.dump(forest, f)

# Load later
with open("forest.pkl", "rb") as f:
    restored = pickle.load(f)

# sample: 330-node forest serializes down to 16kb
```

