import os
import re

_DIR = os.path.dirname(__file__)


def imagenet_labels() -> list[str]:
    """
    The 1001 output classes of Google's inception-2015-12-05 graph: index 0 is the
    background `dummy` class, 1-1000 are ImageNet classes in the graph's node order.

    Built from the vendored official label maps: the pbtxt gives node index ->
    synset, the synset map gives synset -> human name; we keep the first
    comma-separated term of each name.
    """
    with open(os.path.join(_DIR, "imagenet_2012_challenge_label_map_proto.pbtxt")) as f:
        pb = f.read()

    matches = re.findall(
        r'target_class:\s*(\d+)\s*\n\s*target_class_string:\s*"(n\d+)"', pb
    )
    node_to_syn = {int(node): syn for node, syn in matches}

    human = {}
    with open(os.path.join(_DIR, "imagenet_synset_to_human_label_map.txt")) as f:
        for line in f:
            syn, name = line.rstrip("\n").split("\t")
            human[syn] = name.split(",")[0]

    return ["dummy"] + [human[node_to_syn[i]] for i in range(1, 1001)]
