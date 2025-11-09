import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from config import google_cloud_api_key
from matching import (
    build_unique_text_reprs,
    batch_embeddings_with_cache,
    rank_by_edit_dist,
    rank_by_n_gram,
    rank_by_embedding,
    update_stats
)

def graph_results(thresholds:list[float], stats:list[dict], method_name:str):
    accuracies = np.array(
        [(stats[i]["tp"]+stats[i]["tn"])/float(stats[i]["tp"]+stats[i]["fp"]+stats[i]["tn"]+stats[i]["fn"]) 
        for i in range(len(thresholds))]
    )
    #print(accuracies)
    precisions = np.array(
        [stats[i]["tp"]/float(stats[i]["tp"]+stats[i]["fp"]) if stats[i]["tp"] or stats[i]["fp"] else 0 
        for i in range(len(thresholds))]
    )
    #print(precisions)
    recalls = np.array(
        [stats[i]["tp"]/float(stats[i]["tp"]+stats[i]["fn"]) if stats[i]["tp"] or stats[i]["fn"] else 0 
        for i in range(len(thresholds))]
    )
    #print(recalls)
    f1_scores = np.array(
        [precisions[i]*recalls[i]/(precisions[i]+recalls[i]) if precisions[i] or recalls[i] else 0
        for i in range(len(thresholds))]
    )
    #print(f1_scores)
    plt.figure(figsize=(10,6))
    plt.plot(thresholds, accuracies, label="Accuracy", color="blue")
    plt.plot(thresholds, precisions, label="Precision", color="green")
    plt.plot(thresholds, recalls, label="Recall", color="orange")
    plt.plot(thresholds, f1_scores, label="F1 Score", color="purple")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{method_name} Metrics for Different Thresholds")
    plt.legend()
    plt.grid(True)
    # TODO: save under different name with hierarchy
    plt.savefig(f"{method_name}_diff_thresholds_metrics.png")
    plt.show()


if __name__=="__main__":
    file = open("gcmd_ents.json","r")
    gcmd_ents = json.load(file)
    file.close()
    file = open("search_res.json","r")
    wikidata_search_res = json.load(file)
    file.close()
    file = open("ground_truth.json","r")
    ground_truth = json.load(file)
    file.close()
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # adjust as needed
    LABELED_SAMPLES = 475
    # test out different thresholds
    """text_sources, all_texts = build_unique_text_reprs(gcmd_ents, wikidata_search_res, LABELED_SAMPLES)
    all_texts = list(all_texts)
    embeddings = batch_embeddings_with_cache(all_texts, api_key=google_cloud_api_key, db_path="embeddings_cache_old.db")
    text_to_emb = dict(zip(list(all_texts), embeddings))
    gcmd_embeddings = {}
    candidate_embeddings = defaultdict(dict)
    for text, usages in text_sources.items():
        for info in usages:
            if info[0] == "gcmd":
                gcmd_embeddings[info[1]] = text_to_emb[text]
            elif info[0] == "candidate":
                candidate_embeddings[info[1]][info[2]] = text_to_emb[text]"""
    edit_dist_stats = [{"tp":0, "fp":0, "tn":0, "fn":0} for t in thresholds]
    edit_dist_res = {}
    n_gram_stats = [{"tp":0, "fp":0, "tn":0, "fn":0} for t in thresholds]
    n_gram_res = {}
    """embedding_stats = [{"tp":0, "fp":0, "tn":0, "fn":0} for t in thresholds]
    embedding_res = {}"""
    num_samples = 0
    for uuid in gcmd_ents:
        if wikidata_search_res[uuid]:
            ground_truth_matches = ground_truth[uuid].split(",")
            edit_dist_res[uuid] = rank_by_edit_dist(gcmd_ents[uuid]["term"], wikidata_search_res[uuid], threshold=2)
            n_gram_res[uuid] = rank_by_n_gram(gcmd_ents[uuid],wikidata_search_res[uuid], threshold=0)
            """embedding_res[uuid] = [item[0] for item in rank_by_embedding(gcmd_embeddings[uuid],
                                               list(candidate_embeddings[uuid].values()),
                                               list(candidate_embeddings[uuid].keys()))]"""
            for i in range(len(thresholds)):
                edit_dist_res[uuid] = [item for item in edit_dist_res[uuid] if item[1]<=thresholds[len(thresholds)-i-1]]
                update_stats(edit_dist_res[uuid][0][0] if edit_dist_res[uuid] else None, ground_truth_matches, edit_dist_stats[i])
                n_gram_res[uuid] = [item for item in n_gram_res[uuid] if item[1]>=thresholds[i]]
                update_stats(n_gram_res[uuid][0][0] if n_gram_res[uuid] else None, ground_truth_matches, n_gram_stats[i])
                """embedding_res[uuid] = [item for item in embedding_res[uuid] if item[1]>=thresholds[i]]
                update_stats(embedding_res[uuid][0][0] if embedding_res[uuid] else None, ground_truth_matches, embedding_stats[i])"""
        num_samples+=1
        if num_samples==LABELED_SAMPLES:
            break
    graph_results(thresholds, edit_dist_stats, "Edit Distance")
    graph_results(thresholds, n_gram_stats, "N-Gram")