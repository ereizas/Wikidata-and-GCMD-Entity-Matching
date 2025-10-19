import requests
import xml.etree.ElementTree as ET
from json import dump, load
from config import earth_data_user_token, wikidata_access_token
from time import time, sleep

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

def build_path(uuid, data, uuid_to_parent):
    path = []
    current = uuid
    while current in data:
        if uuid!=current:
            path.insert(0, data[current]["term"])
        if current not in uuid_to_parent:
            break
        current = uuid_to_parent[current]
    return "/".join(path)

def write_all_gcmd_ents_to_json():
    """
    Queries the GCMD API using pagination to acquire all of GCMD's entities and their definitions/descriptions
    """
    page = 1
    url = "https://cmr.earthdata.nasa.gov/kms/concepts?page_size=2000&page_num="
    root = None
    HEADERS = {
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {earth_data_user_token}"
    }
    ns = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "dcterms": "http://purl.org/dc/terms/",
            "gcmd": "https://gcmd.earthdata.nasa.gov/kms#"
    }
    response = requests.get(url+f"{page}", headers=HEADERS)
    root = ET.fromstring(response.content)
    num_results = int(root.find(".//gcmd:hits", ns).text)
    filtered_data = dict()
    num_entities = 0
    uuid_to_parent = {}
    while num_entities<num_results:
        if response.status_code==200:
            root = ET.fromstring(response.content)
        else:
            print(response.status_code)
            return
        
        uuid_to_term = {}
        uuid_to_definition = {}
        for concept in root.findall(".//skos:Concept", ns):
            uuid = concept.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")
            if not uuid:
                continue
            label = concept.find("skos:prefLabel", ns)
            if label is not None:
                uuid_to_term[uuid] = label.text.strip()
            definition = concept.find("skos:definition", ns)
            if definition is not None and definition.text is not None:
                uuid_to_definition[uuid] = definition.text.strip()
            broader = concept.find("skos:broader", ns)
            if broader is not None:
                parent_uuid = broader.attrib.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
                uuid_to_parent[uuid] = parent_uuid
            filtered_data[uuid] = {"term":uuid_to_term.get(uuid), "definition":uuid_to_definition.get(uuid)}
            num_entities+=1
        page+=1
        response = requests.get(url+f"{page}", headers=HEADERS)
    # build scheme path after all data is collected in case of split-up parents/children
    for uuid in filtered_data:
        filtered_data[uuid]["path"] = build_path(uuid, filtered_data, uuid_to_parent)
    file = open("gcmd_ents.json","w")
    dump(filtered_data,file)

def get_wikidata_search_results(term:str):
    """
    Queries Wikidata's search engine with a term and returns the search results

    @param term:phrase to search for
    """
    HEADERS = {
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {wikidata_access_token}"
    }
    filtered_data = dict()
    response = requests.get(f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={term}&language=en&format=json&limit=10",
        headers=HEADERS
    )
    data = None
    if response.status_code==200:
        data = response.json()
    else:
        return {"Error occurred":f"{response.status_code}"}
    for res in data["search"]:
        filtered_data[res["id"]] = {"term":res["display"]["label"]["value"],"definition":res["description"] if res.get("description") else None,"match":{"alias":res["match"]["type"],"text":res["match"]["text"]}}
    return filtered_data

def sleep_if_needed(start_time, num_reqs, reqs_per_minute_allowed):
    if num_reqs>=reqs_per_minute_allowed:
        elapsed = time()-start_time
        if elapsed<60:
            sleep(60-elapsed)
        num_reqs = 0
        start_time = time()
    return num_reqs, start_time

def write_search_results_to_json(gcmd_ents_filename):
    """
    Writes the Wikidata search results for each GCMD entity to a JSON file

    @param gcmd_ents_filename
    """
    gcmd_ents = None
    with open(gcmd_ents_filename,'r') as gcmd_ents_file:
        gcmd_ents = load(gcmd_ents_file)
    wiki_data_search_res = None
    with open("gcmd_ents_wikidata_search_res.json",'r') as wiki_data_search_res_file:
        wiki_data_search_res = load(wiki_data_search_res_file)
    REQS_PER_MINUTE_ALLOWED = 5000
    num_reqs = 0
    start_time = None
    for uuid in gcmd_ents:
        if not wiki_data_search_res.get(uuid):
            if start_time is None:
                start_time = time()
            num_reqs, start_time = sleep_if_needed(start_time, num_reqs, REQS_PER_MINUTE_ALLOWED)
            wiki_data_search_res[uuid] = get_wikidata_search_results(gcmd_ents[uuid]["term"])
            num_reqs+=1
        # try with path
        if not wiki_data_search_res.get(uuid) and gcmd_ents[uuid]["path"]:
            path = gcmd_ents[uuid]["path"].split('/')
            if len(path)>=2:
                path = " ".join(path[-2:])
            else:
                path = path[0]
            num_reqs, start_time = sleep_if_needed(start_time, num_reqs, REQS_PER_MINUTE_ALLOWED)
            wiki_data_search_res[uuid] = get_wikidata_search_results(f"{gcmd_ents[uuid]["term"]} {path}")
            num_reqs+=1
    with open("gcmd_ents_wikidata_search_res.json","w") as file:
        dump(wiki_data_search_res,file)

#write_all_gcmd_ents_to_json()
#print(get_wikidata_search_results("carbon"))
#write_search_results_to_json("gcmd_ents.json")

