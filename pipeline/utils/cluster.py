def hcluster(dists, linkage, max_ctm_threshold):
    """
    Perform hierarchical clustering based on pairwise TM scores.

    Args:
        dists:
            [N, N] Pairwise TM scores matrix, where N is the total 
            number of structures in consideration.
        linkage:
            Linkage method for hierarchical clustering (including 
            single, complete, average).
        max_ctm_threshold:
            Maximum TM score threshold between clusters.
    """
  
    def compute_cluster_tm(cluster_i, cluster_j, linkage):
        """
        Compute distance between two clusters based on the input 
        linkage method.
        """

        if linkage == 'single':

            # Closest neighbor (highest tm)
            max_tm = 0
            for i in cluster_i:
                for j in cluster_j:
                    tm = min(dists[i][j], dists[j][i])
                    max_tm = max(max_tm, tm)
            return max_tm

        elif linkage == 'complete':

            # Farthest neighbor (lowest tm)
            min_tm = 1
            for i in cluster_i:
                for j in cluster_j:
                    tm = min(dists[i][j], dists[j][i])
                    min_tm = min(min_tm, tm)
            return min_tm

        else:

            # Average linkage
            sum_tm, count = 0, 0
            for i in cluster_i:
                for j in cluster_j:
                    tm = min(dists[i][j], dists[j][i])
                    sum_tm += tm
                    count += 1
            return sum_tm / count

    # Initilaize
    clusters = [[i] for i in range(dists.shape[0])]
  
    # Perform hierarchical clustering
    while len(clusters) > 1:

        # Find two closest clusters
        cluster_i, cluster_j, max_ctm = None, None, 0
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                ctm = compute_cluster_tm(clusters[i], clusters[j], linkage)
                if ctm > max_ctm:
                    cluster_i, cluster_j, max_ctm = i, j, ctm
    
        # Check for exit
        if max_ctm < max_ctm_threshold:
            break
    
        # Update clusters
        new_cluster = clusters[cluster_i] + clusters[cluster_j]
        del clusters[cluster_j]
        del clusters[cluster_i]
        clusters.append(new_cluster)

    return clusters