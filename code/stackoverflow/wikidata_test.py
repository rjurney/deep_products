import json
import requests


def tuple_to_obj(entity, tag):
    """Take entity/tag url pairs and produce a labeled dict"""
    return {
        'entity': entity.split('/')[-1],
        'entity_url': entity,
        'tag': tag.split('/')[-1],
        'tag_url': tag
    }


url = 'https://query.wikidata.org/sparql'
query = """
# Select all stackexchange tags and their associated entities
SELECT ?type ?value WHERE {
  # type/stackexchange tag/tag value
  ?type wdt:P1482 ?value .
}
"""

r = requests.get(url, params={'format': 'json', 'query': query})
data = r.json()

entity_tag_list = [tuple_to_obj(entity, tag) for entity, tag
                   in [(x['type']['value'], x['value']['value']) for x
                       in data['results']['bindings']]]
entity_tag_mapping = {entity.split('/')[-1]: tag.split('/')[-1] for entity, tag
                      in [(x['type']['value'], x['value']['value']) for x
                          in data['results']['bindings']]}
tag_entity_mapping = {tag.split('/')[-1]: entity.split('/')[-1] for entity, tag
                      in [(x['type']['value'], x['value']['value']) for x
                          in data['results']['bindings']]}


with open('data/stackoverflow/tags/entity_tag_list.jsonl', 'w') as out_f:
    for entity_tag in entity_tag_list:
        out_f.write(json.dumps(entity_tag) + '\n')

with open('data/stackoverflow/tags/entity_tag_mapping.json', 'w') as out_f:
    json.dump(entity_tag_mapping, out_f)

with open('data/stackoverflow/tags/tag_entity_mapping.json', 'w') as out_f:
    json.dump(tag_entity_mapping, out_f)
