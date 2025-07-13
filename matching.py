from nltk import edit_distance
from json import load

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


    