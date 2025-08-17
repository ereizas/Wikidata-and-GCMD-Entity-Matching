from nltk import edit_distance
from json import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#TODO: check the entities where the match has no definition
def format_entity(entity,):
    """
    Formats the entity term and definition into a single string

    @param entity : entity with a term and definition
    @return : single string in lowercase in the format: <term> - <definition>
    """
    return f"{entity["term"]} - {entity["definition"]}".lower()

def match_by_n_gram(target:dict,candidates:dict):
    """
    Creates a vectors for the target entity and the candidates and matches them based on cosine simalarity
    @param target : GCMD entity to match in the format
    @param candidates : potential matches to target from Wikidata
    @return : vector for the target, vectors for the candidates
    """
    target = format_entity(target)
    ids = candidates.keys()
    candidate_texts = [format_entity(candidates[uuid]) for uuid in candidates]
    texts = [target] + candidate_texts
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(texts)
    target_vector = tfidf_matrix[0]
    candidate_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(target_vector, candidate_vectors).flatten()
    ranked = sorted(zip(ids, similarities), key=lambda x: x[1], reverse=True)
    return ranked[0]

def get_best_match(target:str, wikidata_search_res:dict, rank_fxn, inverse:bool=False, threshold:float=None):
    """
    Gets the best match for the target entity from the search results based on the rank function

    @param target : GCMD entity to match
    @param wikidata_search_res : search results on WikiData for the target GCMD entity
    @param rank_fxn : function used for ranking the candidates' match to target
    @param inverse : boolean indicating whether a lower score means better match (inverse order)
    """
    max_score = float("-inf")
    best_match = ""
    for res in wikidata_search_res:
        score = rank_fxn(target.upper(),wikidata_search_res[res]["term"].upper())
        if inverse:
            score = -score
        if ((threshold and score>threshold) or not threshold) and score>max_score:
            max_score = score
            best_match = res
    return best_match

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
    for uuid in gcmd_ents:
        if ground_truth[uuid]:
            print(get_best_match(gcmd_ents[uuid]["term"], wikidata_search_res[uuid], edit_distance, True))
            print(match_by_n_gram(gcmd_ents[uuid],wikidata_search_res[uuid]))
            break
