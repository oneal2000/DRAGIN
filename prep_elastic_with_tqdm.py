from typing import List, Tuple, Union, Dict
import argparse
import glob
import time
import csv
import json
import logging
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader

# Add logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_elasticsearch(
    beir_corpus_file_pattern: str,
    index_name: str,
):
    beir_corpus_files = glob.glob(beir_corpus_file_pattern)
    print(f'#files {len(beir_corpus_files)}')
    
    from beir.retrieval.search.lexical.elastic_search import ElasticSearch
    config = {
        'hostname': 'localhost',
        'index_name': index_name,
        'keys': {'title': 'title', 'body': 'txt'},
        'timeout': 300,  # Increase timeout to 5 minutes
        'retry_on_timeout': True,
        'maxsize': 24,
        'number_of_shards': '1',  # Explicitly set to 1 shard
        'number_of_replicas': '0',  # Set to 0 replicas
        'language': 'english',
    }
    es = ElasticSearch(config)

    # Pre-calculate total number of documents
    total_docs = 0
    for beir_corpus_file in beir_corpus_files:
        with open(beir_corpus_file, 'r') as f:
            total_docs = sum(1 for line in f) - 1  # Subtract header line
    
    logger.info(f"Expected total documents: {total_docs}")

    # create index
    print(f'create index {index_name}')
    try:
        es.delete_index()
        time.sleep(10)  # Increase wait time
        es.create_index()
        time.sleep(5)
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        return

    # Generator with better error handling
    def generate_actions():
        processed_count = 0
        for beir_corpus_file in beir_corpus_files:
            logger.info(f"Processing file: {beir_corpus_file}")
            try:
                with open(beir_corpus_file, 'r', encoding='utf-8') as fin:
                    reader = csv.reader(fin, delimiter='\t')
                    header = next(reader)  # skip header
                    for row_num, row in enumerate(reader, 1):
                        if len(row) >= 3:
                            _id, text, title = row[0], row[1], row[2]
                            es_doc = {
                                '_id': _id,
                                '_op_type': 'index',
                                config['keys']['title']: title,
                                config['keys']['body']: text,
                            }
                            yield es_doc
                            processed_count += 1
                            
                            # Print progress every 100,000 documents
                            if processed_count % 100000 == 0:
                                logger.info(f"Processed {processed_count}/{total_docs} documents")
                        else:
                            logger.warning(f"Skipping malformed row {row_num} in {beir_corpus_file}")
                            
            except Exception as e:
                logger.error(f"Error processing file {beir_corpus_file}: {e}")
                continue

    # Index with progress tracking
    try:
        progress = tqdm(total=total_docs, unit='docs', desc='Indexing')
        es.bulk_add_to_index(generate_actions=generate_actions(), progress=progress)
        progress.close()
        
        # Wait for indexing to complete and refresh
        logger.info("Waiting for indexing to complete...")
        es.es.indices.refresh(index=index_name)
        time.sleep(30)
        
        # Verify final results
        final_count_result = es.es.count(index=index_name)
        final_count = final_count_result['count']
        
        logger.info(f"Indexing completed!")
        logger.info(f"Expected documents: {total_docs}")
        logger.info(f"Indexed documents: {final_count}")
        logger.info(f"Success rate: {(final_count/total_docs)*100:.2f}%")
        
        if final_count < total_docs * 0.99:  # If less than 99% of documents were indexed
            logger.warning(f"Warning: Only {final_count}/{total_docs} documents were indexed!")
            logger.warning("Consider rerunning the indexing process.")
        else:
            logger.info("Indexing completed successfully!")
            
    except Exception as e:
        logger.error(f"Fatal error during indexing: {e}")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='input file')
    parser.add_argument("--index_name", type=str, default=None, help="index name")
    args = parser.parse_args()
    build_elasticsearch(args.data_path, index_name=args.index_name)