from nltk import edit_distance
from json import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def format_entity(entity):
    """
    Formats the entity term and definition into a single string

    @param entity : entity with a term and definition
    @return : single string in lowercase in the format: <term> - <definition>
    """
    return f"{entity["term"]} - {entity["definition"]}".lower()

def rank_by_n_gram(target:dict, candidates:dict):
    """
    Gets the ranking for each candidate based on cosine simalarity with the target
    @param target : GCMD entity to match in the format
    @param candidates : potential matches to target from Wikidata
    @return : vector for the target, vectors for the candidates
    """
    THRESHOLD = 0.044
    target = format_entity(target)
    ids = candidates.keys()
    candidate_texts = [format_entity(candidates[uuid]) for uuid in candidates]
    texts = [target] + candidate_texts
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(texts)
    target_vector = tfidf_matrix[0]
    candidate_vectors = tfidf_matrix[1:]
    similarities = ""
    if 0 not in candidate_vectors.shape:
        similarities = cosine_similarity(target_vector, candidate_vectors).flatten()
        similarities = zip(ids, similarities)
        similarities = [(ent, score) for ent,score in similarities if score>=THRESHOLD]
    return sorted(similarities, key=lambda x: x[1], reverse=True) if type(similarities)!=str else []

def rank_by_edit_dist(target:str, wikidata_search_res:dict, inverse:bool=False):
    """
    Gets the ranking for each candidate based on edit distance from target

    @param target : GCMD entity to match
    @param wikidata_search_res : search results on WikiData for the target GCMD entity
    """
    THRESHOLD = 13
    ranking = []
    for res in wikidata_search_res:
        score = edit_distance(target.upper(),wikidata_search_res[res]["term"].upper())
        if score<THRESHOLD:
            ranking.append((res,score))
    return sorted(ranking, key=lambda x: x[1], reverse=inverse)


def update_stats(rank:list, ground_truth:list, stats:dict):
    """
    Update the statistics needed to calculate accuracy, recall, and precision
    :param rank: list of ids ranked by how well they match the target
    :param ground_truth: list of valid matches
    :param stats: dictionary of statistics (e.g. true positive, false negative)
    """
    if rank and rank[0][0] in ground_truth:
            stats["tp"]+=1
    elif rank and rank[0][0]!=ground_truth:
        stats["fp"]+=1
    elif not rank and ground_truth==[""]:
        stats["tn"]+=1
    elif not rank and ground_truth!=[""]:
        stats["fn"]+=1

def print_performance(method_name:str, stats:dict):
    """
    Prints the performance metrics for a method
    :param method_name: string name of the method
    :param stats: dictionary of statistics (e.g. true positive, false negative)
    """
    print(f"Accuracy of {method_name}: {(stats["tp"]+stats["tn"])/float(stats["tp"]+stats["fp"]+stats["tn"]+stats["fn"])}")
    prec = stats["tp"]/float(stats["tp"]+stats["fp"])
    print(f"Precision of {method_name}: {prec}")
    recall = stats["tp"]/float(stats["tp"]+stats["fn"])
    print(f"Recall of {method_name}: {recall}")
    print(f"F1 score of {method_name}: {prec*recall/(prec+recall)}")

if __name__=="__main__":
    file = open("gcmd_ents.json","r")
    gcmd_ents = load(file)
    file.close()
    file = open("gcmd_ents_wikidata_search_res.json","r")
    wikidata_search_res = load(file)
    file.close()
    file = open("gcmd_ent_wikidata_ent_matching_ground_truth.json","r")
    ground_truth = load(file)
    file.close()
    edit_dist_stats = {"tp":0, "fp":0, "tn":0, "fn":0}
    n_gram_stats = {"tp":0, "fp":0, "tn":0, "fn":0}
    LABELED_SAMPLES = 475
    num_samples = 0
    for uuid in gcmd_ents:
        if wikidata_search_res[uuid]:
            ground_truth_matches = ground_truth[uuid].split(",")
            edit_dist_rank = rank_by_edit_dist(gcmd_ents[uuid]["term"], wikidata_search_res[uuid])
            update_stats(edit_dist_rank, ground_truth_matches, edit_dist_stats)
            n_gram_rank = rank_by_n_gram(gcmd_ents[uuid],wikidata_search_res[uuid])
            update_stats(n_gram_rank, ground_truth_matches, n_gram_stats)
        num_samples+=1
        if num_samples==LABELED_SAMPLES:
            break
    for ind in edit_dist_stats:
        print(f"Edit dist {ind}: {edit_dist_stats[ind]}")
    print("")
    for ind in n_gram_stats:
        print(f"N gram {ind}: {n_gram_stats[ind]}")
    print("")
    print_performance("edit distance", edit_dist_stats)
    print("")
    print_performance("n gram", n_gram_stats)