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

def rank_by_n_gram(target:dict,candidates:dict):
    """
    Gets the ranking for each candidate based on cosine simalarity with the target
    @param target : GCMD entity to match in the format
    @param candidates : potential matches to target from Wikidata
    @return : vector for the target, vectors for the candidates
    """
    THRESHOLD = 0.05
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
        edit_dist_rank = rank_by_edit_dist(gcmd_ents[uuid]["term"], wikidata_search_res[uuid])
        n_gram_rank = rank_by_n_gram(gcmd_ents[uuid],wikidata_search_res[uuid])
        #TODO: parse multiple match entities (in ground_truth[uuid].split(","))
        if edit_dist_rank and edit_dist_rank[0][0]==ground_truth[uuid]:
            edit_dist_stats["tp"]+=1
        elif edit_dist_rank and edit_dist_rank[0][0]!=ground_truth[uuid]:
            edit_dist_stats["fp"]+=1
        elif not edit_dist_rank and ground_truth[uuid]=="":
            edit_dist_stats["tn"]+=1
        elif not edit_dist_rank and ground_truth[uuid]!="":
            edit_dist_stats["fn"]+=1
        if n_gram_rank and n_gram_rank[0][0]==ground_truth[uuid]:
            n_gram_stats["tp"]+=1
        elif n_gram_rank and n_gram_rank[0][0]!=ground_truth[uuid]:
            n_gram_stats["fp"]+=1
        if not n_gram_rank and ground_truth[uuid]=="":
            n_gram_stats["tn"]+=1
        elif not n_gram_rank and ground_truth[uuid]!="":
            n_gram_stats["fn"]+=1  
        num_samples+=1
        if num_samples==LABELED_SAMPLES:
            break
    for ind in edit_dist_stats:
        print(f"Edit dist {ind}: {edit_dist_stats[ind]}")
    print("")
    for ind in n_gram_stats:
        print(f"N gram {ind}: {n_gram_stats[ind]}")
    print("")
    print(f"Accuracy of edit distance: {(edit_dist_stats["tp"]+edit_dist_stats["tn"])/float(edit_dist_stats["tp"]+edit_dist_stats["fp"]+edit_dist_stats["tn"]+edit_dist_stats["fn"])}")
    prec = edit_dist_stats["tp"]/float(edit_dist_stats["tp"]+edit_dist_stats["fp"])
    print(f"Precision of edit distance: {prec}")
    recall = edit_dist_stats["tp"]/float(edit_dist_stats["tp"]+edit_dist_stats["fn"])
    print(f"Recall of edit distance: {recall}")
    print(f"F1 score of edit distance: {prec*recall/(prec+recall)}")
    print("")
    print(f"Accuracy of n gram: {(n_gram_stats["tp"]+n_gram_stats["tn"])/float(n_gram_stats["tp"]+n_gram_stats["fp"]+n_gram_stats["tn"]+n_gram_stats["fn"])}")
    prec = n_gram_stats["tp"]/float(n_gram_stats["tp"]+n_gram_stats["fp"])
    print(f"Precision of n gram: {prec}")
    recall = n_gram_stats["tp"]/float(n_gram_stats["tp"]+n_gram_stats["fn"])
    print(f"Recall of n gram: {recall}")
    print(f"F1 score of n gram: {prec*recall/(prec+recall)}")
