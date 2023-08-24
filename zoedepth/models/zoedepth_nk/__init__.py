from .zoedepth_nk_v1 import ZoeDepthNK

all_versions = {
    "v1": ZoeDepthNK,
}

get_version = lambda v : all_versions[v]