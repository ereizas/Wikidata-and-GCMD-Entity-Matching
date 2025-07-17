from nltk import edit_distance
from json import load
from sklearn.feature_extraction.text import TfidfVectorizer

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

def n_gram_vectorize(ents1:list[str],ents2:list[str]):
    """
    Creates vectors for two lists of formatted entities

    @param ents1 : first list of formatted entities
    @param ents2 : second list of formatted entities
    @return : n-gram vector for ents1, n-gram vector for ents2
    """
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 3))
    all_texts = ents1 + ents2
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    return tfidf_matrix[:len(ents1)], tfidf_matrix[len(ents1):]

def get_best_match(gcmd_ents:dict, wikidata_search_res:dict, target_uuid:str, rank_fxn, inverse:bool=False):
    """
    Gets the best match for the target entity from the search results based on the rank function

    @param gcmd_ents : dictionary of all GCMD entities
    @param wikidata_search_res : search results on WikiData for each GCMD entity
    @param target_uuid : uuid of target entity to match from GCMD
    @param rank_fxn : function used for ranking the candidates' match to target
    @param inverse : boolean indicating whether a lower score means better match (inverse order)
    """
    max_score = float("-inf")
    best_match = ""
    for res in wikidata_search_res[target_uuid]:
        score = rank_fxn(gcmd_ents[target_uuid]["term"],wikidata_search_res[target_uuid][res]["term"])
        if inverse:
            score = -score
        if score>max_score:
            print(f"Score: {score} Term: {wikidata_search_res[target_uuid][res]["term"]}")
            max_score = score
            best_match = res
    return best_match

if __name__=="__main__":
    file = open("gcmd_ents.json","r")
    gcmd_ents = load(file)
    file = open("gcmd_ents_wikidata_search_res.json","r")
    wikidata_search_res = load(file)
    print(get_best_match(gcmd_ents, wikidata_search_res, "b6fd22ab-dca7-4dfa-8812-913453b5695b", edit_distance, True))


    