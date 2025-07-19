from nltk import edit_distance
from json import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def format_entity(entity):
    """
    Formats the entity term and definition into a single string

    @param entity : entity with a term and definition
    @return : single string in lowercase in the format: <term> - <definition>
    """
    return f"{entity["term"]} - {entity["definition"]}".lower()

def format_entities(entities):
    """
    Formats all entities in the given dictionary

    @param entities : dictionary with multiple entities, each with a term and definition
    @return : list of all entities formatted
    """
    return [format_entity(entities[uuid]) for uuid in entities]

def n_gram_vectorize(target:str,candidates:list[str]):
    """
    Creates a vectors for the target entity and the candidates
    @param target : GCMD entity to match in the format: <term> - <definition> 
    @param candidates : list of potential matches from Wikidata in the format: <term> - <definition> 
    @return : vector for the target, vectors for the candidates
    """
    texts = [target] + candidates
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix[0], tfidf_matrix[1:]

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
        score = rank_fxn(target,wikidata_search_res[res]["term"])
        if inverse:
            score = -score
        if ((threshold and score>threshold) or not threshold) and score>max_score:
            print(f"Score: {score} Term: {wikidata_search_res[res]["term"]}")
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
            break
