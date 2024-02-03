from pyserini.search.lucene import LuceneSearcher

print("Import successful")

searcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')

print("Searcher created")

hits = searcher.search("Testing")
print(hits[0].raw)