import requests
from json import dump

def write_all_gcmd_ents_to_json():
    """
    Queries the GCMD API using pagination to acquire all of GCMD's entities and their definitions/descriptions
    """
    page = 1
    url = "https://gcmd.earthdata.nasa.gov/kms/concepts/concept_scheme/sciencekeywords?format=json&page_size=2000&page_num="
    num_results = 0
    response = requests.get(url+f"{page}")
    data = None
    if response.status_code==200:
        data = response.json()
        num_results = data["hits"]
    else:
        print(response.status_code)
        return
    filtered_data = dict()
    num_entities = 0
    while num_entities<num_results:
        if response.status_code==200:
            data = response.json()
            concepts = data.get("concepts")
            if not concepts:
                break
            for concept in data["concepts"]:
                filtered_data[concept["uuid"]] = {"term":concept["prefLabel"],"definition":'\n'.join([d["text"] for d in concept["definitions"]])}
                num_entities+=1
            page+=1
        else:
            print(response.status_code)
            break
        response = requests.get(url+f"{page}") 
    file = open("gcmd_ents.json","w")
    dump(filtered_data,file)
    
def get_wikidata_search_results(term:str):
    """
    Queries Wikidata's search engine with a term and returns the search results

    @param term:phrase to search for
    """
    filtered_data = dict()
    response = requests.get(f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={term}&language=en&format=json&limit=10")
    data = None
    if response.status_code==200:
        data = response.json()
    else:
        return {"Error occurred":f"{response.status_code}"}
    for res in data["search"]:
        filtered_data[res["id"]] = {"term":res["display"]["label"]["value"],"description":res["description"],"match":{"alias":res["match"]["type"],"text":res["match"]["text"]}}
    return filtered_data

#write_all_gcmd_ents_to_json()
print(get_wikidata_search_results("conservation"))
    
