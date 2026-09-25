"""Versioned systematic search protocol; retrieval and eligibility are separate."""
import warnings

SOURCES = {'openalex', 'arxiv', 'ieee', 'scopus'}


def validate_protocol(protocol):
    if protocol.get('schema_version') != 2:
        if 'topics' not in protocol:
            raise ValueError('Expected schema_version: 2 and searches, or legacy topics.')
        warnings.warn('Legacy search YAML: weighted terms no longer exclude papers. Migrate to schema_version: 2.', stacklevel=2)
        return protocol
    searches = protocol.get('searches')
    if not isinstance(searches, list) or not searches:
        raise ValueError('searches must be a nonempty list.')
    ids = set()
    for search in searches:
        identifier = search.get('id')
        if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
            raise ValueError('Each search needs a unique nonempty id.')
        ids.add(identifier)
        queries = search.get('queries')
        if not isinstance(queries, dict) or not queries or set(queries) - SOURCES:
            raise ValueError(f'{identifier}: queries must map supported source names to lists.')
        for source, values in queries.items():
            if not isinstance(values, list) or any(not isinstance(q, str) or not q.strip() for q in values):
                raise ValueError(f'{identifier}/{source}: expected nonempty query strings.')
    criteria = protocol.get('eligibility', {}).get('criteria', [])
    seen = set()
    for criterion in criteria:
        if (not isinstance(criterion.get('id'), str) or not criterion['id'].strip()
                or criterion['id'] in seen or criterion.get('kind') not in {'inclusion', 'exclusion'}
                or not isinstance(criterion.get('description'), str) or not criterion['description'].strip()):
            raise ValueError('Eligibility criteria require unique id, inclusion/exclusion kind, and description.')
        applies = criterion.get('applies_to_search_topics')
        if applies is not None:
            if (not isinstance(applies, list) or not applies
                    or any(not isinstance(topic, str) or not topic.strip() for topic in applies)
                    or len(set(applies)) != len(applies)):
                raise ValueError(
                    f"{criterion['id']}: applies_to_search_topics must be a "
                    "nonempty list of unique search ids."
                )
            unknown = set(applies) - ids
            if unknown:
                raise ValueError(
                    f"{criterion['id']}: unknown applies_to_search_topics: "
                    f"{sorted(unknown)}"
                )
        seen.add(criterion['id'])
    return protocol


def queries_for(protocol, source):
    if protocol.get('schema_version') == 2:
        return [(search['id'], query) for search in protocol['searches']
                for query in search['queries'].get(source, [])]
    return None
