from .zoedepth_v1 import ZoeDepth 

all_versions = {
    "v1": ZoeDepth,
}

get_version = lambda v : all_versions[v]