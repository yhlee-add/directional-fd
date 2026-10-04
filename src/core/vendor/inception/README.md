# Inception 2015-12-05 class labels

The 1001 output classes of Google's `inception-2015-12-05` graph (index 0 is the background `dummy` class, 1-1000 are ImageNet classes in the graph's node order).
clean-fid's `InceptionV3W` is a port of this graph, so these are the labels of its final linear layer's rows; callers pad the list to the graph's padded width of 1008.

Vendored verbatim from the official model tarball
`http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz`:

- `imagenet_2012_challenge_label_map_proto.pbtxt` — node index -> synset
- `imagenet_synset_to_human_label_map.txt` — synset -> human name
- `LICENSE` — the tarball's license

`labels.py` joins the two maps into `list[str]`, keeping the first comma-separated term of each human name (`imagenet_labels()`).
