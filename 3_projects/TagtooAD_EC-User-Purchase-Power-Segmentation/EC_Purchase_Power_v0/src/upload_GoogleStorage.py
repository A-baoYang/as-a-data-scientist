from google.cloud import storage
from google.cloud import bigquery
import os
from io import StringIO


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "application_default_credentials.json"
bq_project = 'gothic-province-823'
bucket_name = 'tagtoo-bigquery-export'
source_file_name = 'test.csv'
destination_blob_name = f'ECAnalysis/upload/{source_file_name}'

    
def upload_StringIO(filename, bq_project, bq_bucket='tagtoo-bigquery-export', bq_filename='abao_upload_test.csv'):
    gcs = storage.Client(project=bq_project)
    gcs.get_bucket(bq_bucket).blob(bq_filename).upload_from_filename(filename, content_type='text/csv')
    
    print('File {} uploaded to {}.'.format(
        bq_filename,
        bq_bucket))
    
def sent_query(
    query,
    project_id,
    dataset_id,
    table_id,
    output,
    ):

    client = bigquery.Client(project=project_id)

    job_config = bigquery.QueryJobConfig()
    table_ref = client.dataset(dataset_id).table(table_id)
    job_config.create_disposition = \
        bigquery.job.CreateDisposition.CREATE_IF_NEEDED
    job_config.write_disposition = \
        bigquery.job.WriteDisposition.WRITE_TRUNCATE
    job_config.destination = table_ref

    query_job = client.query(query, job_config=job_config)

    job_result = query_job.result()  # Wait for the query to finish

    result = {
        'destination_table': table_ref.path,
        'total_rows': job_result.total_rows,
        'total_bytes_processed': query_job.total_bytes_processed,
        'schema': [f.to_api_repr() for f in job_result.schema],
        }

    # Large file: specify a uri including * to shard export
    files = os.path.join(output, '*.csv')
    extract_job = client.extract_table(table_ref, files)
    extract_job.result()  # Wait for export to finish
    with open('raw_data_gcs_path.txt', 'w') as f:
        f.write(files)

    print(f'file has stored to {files}.')
    return result

    