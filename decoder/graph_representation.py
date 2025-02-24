import numpy as np
import torch
import scipy

def sample_syndromes(n_shots, compiled_sampler, device):
    # distinguish between training and testing:
    if compiled_sampler.__class__ == list:
        # sample for each error rate:
        n_trivial_syndromes = 0
        detection_events_list, observable_flips_list = [], []
        n_shots_one_p = n_shots // len(compiled_sampler)
        for sampler in compiled_sampler:
            # repeat each experiments multiple times to get enough non-empty:
            detections_one_p, observable_flips_one_p = [], []
            while len(detections_one_p) < n_shots_one_p:
                detection_events, observable_flips = sampler.sample(
                shots=n_shots_one_p * 10,
                separate_observables=True)
                # sums over the detectors to check if we have a parity change
                shots_w_flips = np.sum(detection_events, axis=1) != 0
                # save only data for measurements with non-empty syndromes
                # but count how many trivial (identity) syndromes we have
                n_trivial_syndromes += np.invert(shots_w_flips).sum()
                detections_one_p.extend(detection_events[shots_w_flips, :])
                observable_flips_one_p.extend(observable_flips[shots_w_flips, :])
            # if there are more non-empty syndromes than necessary
            detection_events_list.append(detections_one_p[:n_shots_one_p])
            observable_flips_list.append(observable_flips_one_p[:n_shots_one_p])
    else:
        detection_events_list, observable_flips_list = compiled_sampler.sample(
            shots=n_shots,
            separate_observables=True)
        # sums over the detectors to check if we have a parity change
        shots_w_flips = np.sum(detection_events_list, axis=1) != 0
        # save only data for measurements with non-empty syndromes
        # but count how many trivial (identity) syndromes we have
        n_trivial_syndromes = np.invert(shots_w_flips).sum()
        detection_events_list = detection_events_list[shots_w_flips, :]
        observable_flips_list = observable_flips_list[shots_w_flips, :]

    # make an array from the list:
    detection_events = np.array(detection_events_list)
    observable_flips = np.array(observable_flips_list)
    observable_flips = torch.tensor(observable_flips, dtype=torch.float32).to(device)
    return detection_events, observable_flips, n_trivial_syndromes

def get_batch_of_node_features(syndrome_3D):
    # syndromes come in shape [n_shots, x_coordinate, z_coordinate, time]
    # get the nonzero entries (node features):
    defect_inds = np.nonzero(syndrome_3D)
    node_features = np.transpose(np.array(defect_inds)).astype(np.float32)
    batch_labels = node_features[:, 0].astype(np.int64)
    # exclude batch index from node features: 
    return node_features[:, 1:], batch_labels
    
def get_batch_of_edges(node_features, batch_labels, device):
    # reshape the batch labels to shape (n, m), where there are n points of dimension m
    batch_labels = batch_labels.reshape(-1, 1)
    tree = scipy.spatial.cKDTree(batch_labels)
    # get the edge list (comes in list of tuples, convert to array):
    # works by finding all neighbours that are closer than some number bigger than 0, 
    # i.e. all neighbours that are in the same batch
    # note: gives a directed, fully connected graph without self-loops
    edge_index = np.array(list(tree.query_pairs(r = 0.2)), dtype=np.int64)
    # reshape to form (n_edges, 2) to get right dimension if just one edge present:
    edge_index = edge_index.reshape(-1, 2)
    # make the graph undirected:
    edge_index = np.vstack((edge_index, edge_index[:, [1, 0]]))
    # sort by first (row) index: 
    edge_index = edge_index[np.argsort(edge_index[:, 0])]

    # compute the distances between the nodes (start node - end node):
    x_dist = np.abs(node_features[edge_index[:, 1], 0] - 
                    node_features[edge_index[:, 0], 0])

    y_dist = np.abs(node_features[edge_index[:, 1], 1] - 
                    node_features[edge_index[:, 0], 1])

    t_dist = np.abs(node_features[edge_index[:, 1], 2] - 
        node_features[edge_index[:, 0], 2])

    # inverse square of the supremum norm between two nodes
    edge_attr = np.maximum.reduce([y_dist, x_dist, t_dist])
    edge_attr = 1 / edge_attr ** 2
    # convert to torch tensors:
    edge_attr = torch.from_numpy(edge_attr).to(device)
    edge_index = torch.from_numpy(edge_index).to(device)

    return edge_index.T, edge_attr